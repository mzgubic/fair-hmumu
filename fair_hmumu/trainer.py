import os
import time
import numpy as np
import itertools
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from fair_hmumu import defs
from fair_hmumu import plot
from fair_hmumu import models
from fair_hmumu import utils
from fair_hmumu import dataset
from fair_hmumu.utils import timeit
from fair_hmumu.preprocessing import PCAWhiteningPreprocessor
from fair_hmumu.preprocessing import OutputTransformer
from fair_hmumu.environment import TFEnvironment


class Master:

    def __init__(self, run_conf):

        # configurations
        self.loc = run_conf.loc
        self.clf = models.Classifier(run_conf.get('Classifier'))
        self.adv = models.Adversary.create(run_conf.get('Adversary'))
        self.run_conf = run_conf
        self.opt_conf = run_conf.get('Optimiser')
        self.trn_conf = run_conf.get('Training')
        self.bcm_conf = run_conf.get('Benchmark')
        self.plt_conf = run_conf.get('Plotting')

        print('------------')
        print('--- Settings:')
        print(self.clf)
        print(self.adv)
        print('Optimisation', self.opt_conf)
        print('Training', self.trn_conf)
        print('Benchmark', self.bcm_conf)
        print('Plotting', self.plt_conf)
        print('------------')

        # load the data handler and the data
        self.njet = self.trn_conf['njet']
        self.dh = None
        self.bcm_features = dataset.feature_list(self.trn_conf['bcm_features'], self.njet)
        self.clf_features = dataset.feature_list(self.trn_conf['clf_features'], self.njet)
        self._ds = {}
        self._tt = ['train', 'test']
        self._tts = ['train', 'test', 'ss']
        self._load_data()

        # prepare losses
        self._losses = {tt:{n:[] for n in ['C', 'A', 'CA', 'BCM']} for tt in self._tt}

        # prepare metrics
        self.metrics = ['roc_auc', 'sig_eff', 'ks_metric']
        self._metric_vals = {met:{tt:[] for tt in self._tt} for met in self.metrics}

        # prepare classifier scores
        self.bcm = None
        self.bcm_score = {}
        self.clf_score = {}
        self.score_loc = utils.makedir(os.path.join(self.loc, 'clf_scores'.format(self.clf.name)))
        
    def _load_data(self):

        # data handling
        self.dh = dataset.DatasetHandler(self.trn_conf, seed=42)
        
        # load
        self._ds['train'] = self.dh.get_train(self.clf_features)
        self._ds['test'] = self.dh.get_test(self.clf_features)
        self._ds['ss'] = self.dh.get_ss(self.clf_features)

    def _fit_preprocessing(self):

        self.pre = {}
        for xz in ['X', 'Z']:

            # fit the data preprocessing for the features and the mass
            self.pre[xz] = PCAWhiteningPreprocessor(n_cpts=self._ds['train'][xz].shape[1])
            self.pre[xz].fit(self._ds['train'][xz])
            self.pre[xz].save(os.path.join(self.loc, 'PCA_{}.pkl'.format(xz)))

            # apply it to the datasets
            for tts in self._tts:
                self._ds[tts][xz] = self.pre[xz].transform(self._ds[tts][xz])

    def _load_preprocessing(self):

        self.pre = {}
        for xz in ['X', 'Z']:

            # fit the data preprocessing for the features and the mass
            self.pre[xz] = PCAWhiteningPreprocessor.from_file(os.path.join(self.loc, 'PCA_{}.pkl'.format(xz)))

            # apply it to the datasets
            for tts in self._tts:
                self._ds[tts][xz] = self.pre[xz].transform(self._ds[tts][xz])

    def _setup_environment(self):

        # set up the tensorflow environment in which the graphs are built and executed
        self.env = TFEnvironment(self.clf, self.adv, self.opt_conf)
        self.env.build(self.preprocess(self.dh.get_batch(self.clf_features)))
        self.env.initialise_variables()

    def preprocess(self, batch):

        # transform X and Z
        for xz in ['X', 'Z']:
            batch[xz] = self.pre[xz].transform(batch[xz])

        return batch


class Trainer(Master):

    def __init__(self, run_conf):

        # convenience
        super().__init__(run_conf)

        # data preprocessing
        self._fit_preprocessing()

        # set up TensorFlow environment
        self._setup_environment()

    def pretrain(self):

        # benchmark training
        self._train_benchmarks()
        self._predict_benchmarks()

        # classifier training
        self._pretrain_classifier()

    @timeit
    def _pretrain_classifier(self):

        print('--- Pretraining for {} steps'.format(self.trn_conf['n_pre']))

        # pretrain the classifier
        for _ in range(self.trn_conf['n_pre']):
            batch = self.dh.get_batch(self.clf_features)
            batch = self.preprocess(batch)
            self.env.pretrain_step(batch)
            self._track_losses()

        # pretrain the adversary
        for _ in range(self.trn_conf['n_pre']):
            batch = self.dh.get_batch(self.clf_features)
            batch = self.preprocess(batch)
            self.env.train_step_adv(batch)
            self._track_losses()

    @timeit
    def _train_benchmarks(self):

        print('--- Training the benchmark {} model'.format(self.bcm_conf['type']))

        # construct hyperparameters
        bcm_hps = dict(self.bcm_conf)
        bcm_hps.pop('type')

        # instantiate the model
        if self.bcm_conf['type'] == 'GBC':
            self.bcm = GradientBoostingClassifier(**bcm_hps)
        if self.bcm_conf['type'] == 'XGB':
            self.bcm = XGBClassifier(**bcm_hps)

        # training set is very unbalanced, fit without the weights
        # do not apply preprocessing to the benchmark
        train = self.dh.get_train(self.bcm_features)
        self.bcm.fit(train['X'], train['Y'].ravel())

    @timeit
    def _predict_benchmarks(self):

        print('--- Making benchmark prediction on the test and ss events')

        # predict on the test set
        ds, pred, label, weight = {}, {}, {}, {}
        ds['ss'] = self.dh.get_ss(features=self.bcm_features)
        ds['test'] = self.dh.get_test(features=self.bcm_features)
        ds['train'] = self.dh.get_train(features=self.bcm_features)
        pred['ss'] = self.bcm.predict_proba(ds['ss']['X'])[:, 1].reshape(-1, 1)

        for tt in self._tt:

            # predictions, labels, weights
            pred[tt] = self.bcm.predict_proba(ds[tt]['X'])[:, 1].reshape(-1, 1)
            label[tt] = ds[tt]['Y']
            weight[tt] = ds[tt]['W']

            # and store the score
            name = '{}_{}'.format(self.bcm_conf['type'], tt)
            self.bcm_score[tt] = self._get_clf_score(name, pred[tt], label[tt], weight[tt], pred['ss'])
            self.bcm_score[tt].save(os.path.join(self.score_loc, self.bcm_score[tt].fname))

            # save the loss as well (https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits)
            loss = np.mean(- label[tt] * np.log(pred[tt]) - (1-label[tt]) * np.log(1-pred[tt]))
            self._losses[tt]['BCM'] = loss

    @timeit
    def train(self):

        print('--- Training for {} steps'.format(self.trn_conf['n_epochs']))

        n_epochs = self.trn_conf['n_epochs']
        t0 = time.time()

        for istep in range(n_epochs):

            # train the classifier
            for _ in range(self.trn_conf['n_clf']):
                batch = self.dh.get_batch(self.clf_features)
                batch = self.preprocess(batch)
                self.env.train_step_clf(batch)

            # train the adversary
            for _ in range(self.trn_conf['n_adv']):
                batch = self.dh.get_batch(self.clf_features)
                batch = self.preprocess(batch)
                self.env.train_step_adv(batch)

            # track losses (cheap) and metrics (expensive, skip n_skip)
            self._track_losses()
            self._track_metrics()

            # make plots
            is_final_step = (istep == n_epochs-1)
            if is_final_step or istep%self.plt_conf['n_skip'] == 0:
                self._assess_scores(istep)
                self._make_plots(istep, is_final_step)
                t1 = time.time()
                print('{} steps took {:2.2f}s. Time left to train: {:2.2f}min'.format(istep, t1-t0, (n_epochs-istep)*(t1-t0)/(istep+0.01)/60.))

        # transform the output to a uniform distribution
        self._train_output_transform()

        # write a bash script that can be run to make gifs
        self._gif()
        self._record_summary()

    def _track_losses(self):

        for tt in self._tt:
            C, A, CA = self.env.losses(self._ds[tt])
            self._losses[tt]['C'].append(C)
            self._losses[tt]['A'].append(A)
            self._losses[tt]['CA'].append(CA)

    def _track_metrics(self):

        for tt, metric in itertools.product(self._tt, self.metrics):
            try:
                self._metric_vals[metric][tt].append(self.clf_score[tt][metric])
            except KeyError:
                pass

    def _assess_scores(self, istep):

        # predict on ss ds
        pred = {}
        pred['ss'] = self.env.clf_predict(self._ds['ss'])

        # assess scores
        for tt in self._tt:
            pred[tt] = self.env.clf_predict(self._ds[tt])

            # and construct score objects
            name = '{}_{}_{}'.format(self.clf.name, tt, istep)
            self.clf_score[tt] = self._get_clf_score(name, pred[tt], self._ds[tt]['Y'], self._ds[tt]['W'], pred['ss'])
            self.clf_score[tt].save(os.path.join(self.score_loc, self.clf_score[tt].fname))

    def _get_clf_score(self, name, pred, label, weight, ss_pred):

        # roc curves
        roc = self._get_roc(pred, label, weight)

        # clf distros for different mass ranges
        clf_hists = self._get_clf_hists(ss_pred)

        # mass distros for different clf percentiles
        mass_hists = self._get_mass_hists(ss_pred)

        # return the score object
        return ClassifierScore(name, roc, clf_hists, mass_hists)

    def _get_roc(self, pred, label, weight):

        # get roc curve
        fprs, tprs, _ = sklearn.metrics.roc_curve(label, pred.ravel(), sample_weight=weight)
        roc_auc = sklearn.metrics.roc_auc_score(label, pred.ravel(), sample_weight=weight)

        # compute the signal efficiency at 99% background rejection
        # WARNING: need to have monotonically increasing inputs to np.interp, otherwise results are rubbish
        sig_eff = np.interp(1-0.99, fprs, tprs)

        # compress the roc curve to N points (not a million)
        keep = list(np.linspace(0, len(fprs)-1, 1000, dtype=int))
        fprs = fprs[keep]
        tprs = tprs[keep]

        return fprs, tprs, roc_auc, sig_eff

    def _get_clf_hists(self, ss_pred):

        # determine the indices of events in a mass range
        mass = self.pre['Z'].inverse_transform(self._ds['ss']['Z'])
        ind = {}
        ind['low'] = mass < 120
        ind['high'] = mass > 130
        ind['medium'] = np.logical_and(np.logical_not(ind['low']), np.logical_not(ind['high']))

        # now compute the classifier distribution
        clf_hists = {}
        for mass_range in ind:
            clf_outputs = ss_pred[ind[mass_range]]
            weights = self._ds['ss']['W'][ind[mass_range]]
            clf_hists[mass_range], _ = np.histogram(clf_outputs, weights=weights, bins=defs.bins, range=(0, 1), density=True)

        return clf_hists

    def _get_mass_hists(self, ss_pred):

        # get real mass values
        mass = self.pre['Z'].inverse_transform(self._ds['ss']['Z'])

        mass_hists = {}
        for perc in self.plt_conf['percentiles']:

            # determine the mass values in top percentile
            cut = np.percentile(ss_pred, 100-perc)
            sel_mass = mass[ss_pred > cut]
            sel_weights = self._ds['ss']['W'][ss_pred > cut]

            # and compute histograms
            sel_hist, _ = np.histogram(sel_mass, weights=sel_weights, bins=defs.bins, range=(defs.mlow, defs.mhigh))
            full_hist, _ = np.histogram(mass, weights=self._ds['ss']['W'], bins=defs.bins, range=(defs.mlow, defs.mhigh))

            # pack up in dict
            mass_hists[perc] = sel_hist, full_hist

        return mass_hists

    def _make_plots(self, istep, is_final_step):

        # get the plotting summaries
        bcm_plot, clf_plot = {}, {}

        for tt in self._tt:

            # plot setup
            lstyles = {'test':'-', 'train':':'}

            bcm_plot[tt] = {'score':self.bcm_score[tt],
                            'label':'{} ({})'.format(self.bcm_conf['type'], tt),
                            'colour':defs.dark_blue,
                            'style':lstyles[tt]}

            clf_plot[tt] = {'score':self.clf_score[tt],
                            'label':'{} ({})'.format(self.clf.name, tt),
                            'colour':defs.blue,
                            'style':lstyles[tt]}

        # package them up
        all_setups = [plot[tt] for plot in [bcm_plot, clf_plot] for tt in self._tt]
        test_setups = [plot['test'] for plot in [bcm_plot, clf_plot]]

        # determine the unique id and location of the plot
        unique_id = 'final' if is_final_step else '{:04d}'.format(istep)
        def loc(name):
            return self.loc if is_final_step else utils.makedir(os.path.join(self.loc, name))

        # loss plot
        plot.losses(self._losses, self.run_conf, loc('losses'), unique_id)

        # metric plot
        plot.metrics(self._metric_vals, self.bcm_score, self.run_conf, loc('metric'), unique_id)

        # roc plot
        plot.roc_curve(all_setups, self.run_conf, loc('roc_curve'), unique_id, zoom=True)
        plot.roc_curve(all_setups, self.run_conf, loc('roc_curve'), unique_id, zoom=False)

        # clf output plot
        plot.clf_output(test_setups, self.run_conf, loc('clf_output'), unique_id)

        # mass distro plots
        for perc in self.plt_conf['percentiles']:
            pname = 'mass_shape_{}p'.format(perc)
            plot.mass_shape(test_setups, perc, self.run_conf, loc(pname), unique_id)

        # KS test plots
        plot.KS_test(test_setups, self.run_conf, loc('KS_test'), unique_id)

    def _train_output_transform(self):

        # transform prediction to a uniform distribution
        predictions = self.env.clf_predict(self._ds['ss'])

        # train on the spurious signal events as they are the most abundant
        self.out_tsf = OutputTransformer(n_quantiles=1000, output_distribution='uniform')
        self.out_tsf.fit(predictions)

        # save the transformer
        n_div = self.trn_conf['n_div']
        n_rmd = self.trn_conf['n_rmd']
        self.out_tsf.save(os.path.join(self.loc, 'QuantileTransform.pkl'))

    def _record_summary(self):

        def write_number(number, fname):
            with open(os.path.join(self.loc, fname), 'w') as f:
                f.write('{:2.4f}\n'.format(number))

        # loop over metrics and test/train combinations
        for metric, tt in itertools.product(self.metrics, self._tt):

            # write benchmark and classifier values
            write_number(self.clf_score[tt][metric], 'metric__clf__{}__{}.txt'.format(tt, metric))
            write_number(self.bcm_score[tt][metric], 'metric__bcm__{}__{}.txt'.format(tt, metric))

    def _gif(self):

        gifs = ['roc_curve', 'clf_output', 'losses']
        gifs += ['mass_shape_{}p'.format(p) for p in self.plt_conf['percentiles']]

        script = os.path.join(self.loc, 'make_gifs.sh')

        with open(script, 'w') as f:
            for gif in gifs:

                # convert frames to png
                f.write('mogrify -density 100 -format png {}/*.pdf\n'.format(gif))

                # make the gif
                png_loc = os.path.join(self.loc, gif)
                pngs = ' '.join([os.path.join(png_loc, fname[:-4]+'.png') for fname in os.listdir(png_loc)])
                gif_path = os.path.join(self.loc, '{}.gif'.format(gif))
                f.write('convert -colors 32 -loop 0 -delay 10 {i} {o}\n'.format(i=pngs, o=gif_path))

                # rm pngs to save disk space
                f.write('rm {}/*.png\n\n'.format(gif))

    def save_model(self):
        
        n_div = self.trn_conf['n_div']
        n_rmd = self.trn_conf['n_rmd']
        path = os.path.join(self.loc, 'Classifier.pkl')
        self.env.save_model(path)


class Predictor(Master):

    def __init__(self, run_conf):

        # convenience
        run_conf.set('Training', 'entrystop', -1)
        super().__init__(run_conf)

        # data preprocessing
        self._load_preprocessing()

        # set up TensorFlow environment
        self._setup_environment()

    def load_model(self):

        # load model
        n_div = self.trn_conf['n_div']
        n_rmd = self.trn_conf['n_rmd']
        path = os.path.join(self.loc, 'Classifier.pkl')
        self.env.load_model(path)
        
        # and the output transform
        self._load_output_transform()

    def _load_output_transform(self):

        # train on the spurious signal events as they are the most abundant
        path = os.path.join(self.loc, 'QuantileTransform.pkl')
        self.out_tsf = OutputTransformer.from_file(path)

    def predict(self, data):

        # predict using the model
        pred = self.env.clf_predict(data)

        # transform the output
        output = self.out_tsf(pred)

        return output


class ClassifierScore(utils.Saveable):

    def __init__(self, name, roc, clf_hists, mass_hists):

        super().__init__(name)
        self.fname = '{}_score.pkl'.format(self.name)
        self.roc_curve = roc[:2]
        self.roc_auc = roc[2]
        self.sig_eff = roc[3]
        self.clf_hists = clf_hists
        self.mass_hists = mass_hists
        self.ks_metric = 0

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):

        return '{} score'.format(self.name)






