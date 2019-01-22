import os
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from fair_hmumu import defs
from fair_hmumu import plot
from fair_hmumu import models
from fair_hmumu import utils
from fair_hmumu.dataset import DatasetHandler
from fair_hmumu.preprocessing import PCAWhiteningPreprocessor
from fair_hmumu.environment import TFEnvironment


class Trainer:

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
        self._ds = {}
        self._tt = ['train', 'test']
        self._tts = ['train', 'test', 'ss']
        self._load_data()

        # prepare losses

        self._losses = {tt:{n:[] for n in ['C', 'A', 'CA', 'BCM']} for tt in self._tt}

        # prepare classifier scores
        self.bcm = None
        self.bcm_score = {}
        self.clf_score = {}
        self.score_loc = utils.makedir(os.path.join(self.loc, 'clf_scores'.format(self.clf.name)))

        # data preprocessing
        self._fit_preprocessing()

        # set up TensorFlow environment
        self._setup_environment()



    def _load_data(self):

        # data handling
        production = self.trn_conf['production']
        features = ['Z_PT', 'Muons_CosThetaStar']
        features = ['Muons_Eta_Lead', 'Muons_Eta_Sub', 'Muons_Phi_Lead', 'Muons_Phi_Sub', 'Muons_PT_Lead', 'Muons_PT_Sub']
        entrystop = self.trn_conf['entrystop']
        self.dh = DatasetHandler(production, features, entrystop=entrystop, test_frac=0.25, seed=42)

        # load
        self._ds['train'] = self.dh.get_train(defs.jet0) # TODO: jet channels
        self._ds['test'] = self.dh.get_test(defs.jet0)
        self._ds['ss'] = self.dh.get_ss(defs.jet0)

    def _fit_preprocessing(self):

        self.pre = {}
        for xz in ['X', 'Z']:

            # fit the data preprocessing for the features and the mass
            self.pre[xz] = PCAWhiteningPreprocessor(n_cpts=self._ds['train'][xz].shape[1])
            self.pre[xz].fit(self._ds['train'][xz])
            self.pre[xz].save(os.path.join(self.loc, 'PCA_{}_{}.pkl'.format(xz, defs.jet0)))

            # apply it to the datasets
            for tts in self._tts:
                self._ds[tts][xz] = self.pre[xz].transform(self._ds[tts][xz])

    def _setup_environment(self):

        # set up the tensorflow environment in which the graphs are built and executed
        self.env = TFEnvironment(self.clf, self.adv, self.opt_conf)
        self.env.build(self.transform(self.dh.get_batch(defs.jet0)))
        self.env.initialise_variables()

    def transform(self, batch):

        # transform X and Z
        for xz in ['X', 'Z']:
            batch[xz] = self.pre[xz].transform(batch[xz])

        return batch

    def pretrain(self):

        # benchmark training
        self._train_benchmarks()
        self._predict_benchmarks()

        # classifier training
        self._pretrain_classifier()

    def _pretrain_classifier(self):

        print('--- Pretraining for {} steps'.format(self.trn_conf['n_pre']))

        # pretrain the classifier
        for _ in range(self.trn_conf['n_pre']):
            batch = self.dh.get_batch(defs.jet0)
            batch = self.transform(batch)
            self.env.pretrain_step(batch)
            self._assess_losses()

        # pretrain the adversary
        for _ in range(self.trn_conf['n_pre']):
            batch = self.dh.get_batch(defs.jet0)
            batch = self.transform(batch)
            self.env.train_step_adv(batch)
            self._assess_losses()

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
        self.bcm.fit(self._ds['train']['X'], self._ds['train']['Y'].ravel())

    def _predict_benchmarks(self):

        print('--- Making benchmark prediction on the test and ss events')

        # predict on the test set
        pred, label, weight = {}, {}, {}
        pred['ss'] = self.bcm.predict_proba(self._ds['ss']['X'])[:, 1].reshape(-1, 1)

        for tt in self._tt:

            pred[tt] = self.bcm.predict_proba(self._ds[tt]['X'])[:, 1].reshape(-1, 1)
            label[tt] = self._ds[tt]['Y']
            weight[tt] = self._ds[tt]['W']

            # and store the score
            name = '{}_{}'.format(self.bcm_conf['type'], tt)
            self.bcm_score[tt] = self._get_clf_score(name, pred[tt], label[tt], weight[tt], pred['ss'])
            self.bcm_score[tt].save(os.path.join(self.score_loc, self.bcm_score[tt].fname))

            # save the loss as well (https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits)
            loss = np.mean(- label[tt] * np.log(pred[tt]) - (1-label[tt]) * np.log(1-pred[tt]))
            self._losses[tt]['BCM'] = loss

    def train(self):

        print('--- Training for {} steps'.format(self.trn_conf['n_epochs']))

        n_epochs = self.trn_conf['n_epochs']

        for istep in range(n_epochs):

            # train the classifier
            for _ in range(self.trn_conf['n_clf']):
                batch = self.dh.get_batch(defs.jet0)
                batch = self.transform(batch)
                self.env.train_step_clf(batch)

            # train the adversary
            for _ in range(self.trn_conf['n_adv']):
                batch = self.dh.get_batch(defs.jet0)
                batch = self.transform(batch)
                self.env.train_step_adv(batch)

            # track losses (cheap)
            self._assess_losses()

            # make plots
            is_final_step = (istep == n_epochs-1)
            if is_final_step or istep%self.plt_conf['n_skip'] == 0:
                self._assess_scores(istep)
                self._make_plots(istep, is_final_step)

        # write a bash script that can be run to make gifs
        self._gif()
        self._record_summary()

    def _assess_losses(self):

        for tt in self._tt:
            C, A, CA = self.env.losses(self._ds[tt])
            self._losses[tt]['C'].append(C)
            self._losses[tt]['A'].append(A)
            self._losses[tt]['CA'].append(CA)

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

        fprs, tprs, _ = sklearn.metrics.roc_curve(label, pred.ravel(), sample_weight=weight)
        roc_auc = sklearn.metrics.roc_auc_score(label, pred.ravel(), sample_weight=weight)

        return fprs, tprs, roc_auc

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

        # roc plot
        plot.roc_curve(all_setups, self.run_conf, loc('roc_curve'), unique_id)

        # clf output plot
        plot.clf_output(test_setups, self.run_conf, loc('clf_output'), unique_id)

        # mass distro plots
        for perc in self.plt_conf['percentiles']:
            pname = 'mass_shape_{}p'.format(perc)
            plot.mass_shape(test_setups, perc, self.run_conf, loc(pname), unique_id)

        # KS test plots
        plot.KS_test(test_setups, self.run_conf, loc('KS_test'), unique_id)


    def _record_summary(self):

        def write_number(number, fname):
            with open(os.path.join(self.loc, fname), 'w') as f:
                f.write('{:2.4f}\n'.format(number))

        # roc auc
        write_number(self.clf_score['test'].roc_auc, 'roc_auc_clf.txt')
        write_number(self.bcm_score['test'].roc_auc, 'roc_auc_bcm.txt')

        # ks test
        write_number(self.clf_score['test'].ks_metric, 'ks_metric_clf.txt')
        write_number(self.bcm_score['test'].ks_metric, 'ks_metric_bcm.txt')

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


class ClassifierScore(utils.Saveable):

    def __init__(self, name, roc, clf_hists, mass_hists):

        super().__init__(name)
        self.fname = '{}_score.pkl'.format(self.name)
        self.roc_curve = roc[:2]
        self.roc_auc = roc[2]
        self.clf_hists = clf_hists
        self.mass_hists = mass_hists
        self.ks_metric = 0

    def __str__(self):

        return '{}: ROC AUC=0.5, fairness=0.5'.format(self.name)






