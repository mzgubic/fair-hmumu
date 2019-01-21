import os
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
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
        self.opt_conf = run_conf.get('Optimiser')
        self.trn_conf = run_conf.get('Training')
        self.bcm_conf = run_conf.get('Benchmark')
        self.plt_conf = run_conf.get('Plotting')
        self.percentiles = [50, 10]

        print('------------')
        print('--- Settings:')
        print(self.clf)
        print(self.adv)
        print(self.opt_conf)
        print(self.trn_conf)
        print(self.bcm_conf)
        print('------------')

        # load the data handler and the data
        self._load_data()

        # data preprocessing
        self.bcm = None
        self.bcm_test_score = None
        self.bcm_train_score = None
        self.score_loc = utils.makedir(os.path.join(self.loc, 'clf_scores'.format(self.clf.name)))
        self._fit_preprocessing()

        # set up TensorFlow environment
        self._setup_environment()

        # prepare losses
        self._losses = {n:[] for n in ['C', 'A', 'CA']}

    def _load_data(self):

        # data handling
        production = self.trn_conf['production']
        features = ['Z_PT', 'Muons_CosThetaStar']
        features = ['Muons_Eta_Lead', 'Muons_Eta_Sub', 'Muons_Phi_Lead', 'Muons_Phi_Sub', 'Muons_PT_Lead', 'Muons_PT_Sub']
        entrystop = self.trn_conf['entrystop']
        self.dh = DatasetHandler(production, features, entrystop=entrystop, test_frac=0.25, seed=42)

        # load
        self._train = self.dh.get_train(defs.jet0) # TODO: jet channels
        self._test = self.dh.get_test(defs.jet0)
        self._ss = self.dh.get_ss(defs.jet0)

    def _fit_preprocessing(self):

        self.pre = {}
        for xz in ['X', 'Z']:

            # fit the data preprocessing for the features and the mass
            self.pre[xz] = PCAWhiteningPreprocessor(n_cpts=self._train[xz].shape[1])
            self.pre[xz].fit(self._train[xz])
            self.pre[xz].save(os.path.join(self.loc, 'PCA_{}_{}.pkl'.format(xz, defs.jet0)))

            # apply it to the datasets
            self._train[xz] = self.pre[xz].transform(self._train[xz])
            self._test[xz] = self.pre[xz].transform(self._test[xz])
            self._ss[xz] = self.pre[xz].transform(self._ss[xz])

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

        # training set is very unbalanced, fit without the weights
        self.bcm.fit(self._train['X'], self._train['Y'].ravel())

    def _predict_benchmarks(self):

        print('--- Making benchmark prediction on the test and ss events')

        # predict on the test set
        test_pred = self.bcm.predict_proba(self._test['X'])[:, 1].reshape(-1, 1)
        test_label = self._test['Y']
        test_weight = self._test['W']
        train_pred = self.bcm.predict_proba(self._train['X'])[:, 1].reshape(-1, 1)
        train_label = self._train['Y']
        train_weight = self._train['W']
        ss_pred = self.bcm.predict_proba(self._ss['X'])[:, 1].reshape(-1, 1)

        # and store the score
        test_name = '{}_test'.format(self.bcm_conf['type'])
        self.bcm_test_score = self.assess_clf(test_name, test_pred, test_label, test_weight, ss_pred)
        self.bcm_test_score.save(os.path.join(self.score_loc, self.bcm_test_score.fname))
        train_name = '{}_train'.format(self.bcm_conf['type'])
        self.bcm_train_score = self.assess_clf(train_name, train_pred, train_label, train_weight, ss_pred)
        self.bcm_train_score.save(os.path.join(self.score_loc, self.bcm_train_score.fname))

        # save the loss as well (https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits)
        labels = self._test['Y'].ravel()
        preds = test_pred.ravel()
        loss = - labels * np.log(preds) - (1-labels) * np.log(1-preds)
        self._losses['BCM'] = np.mean(loss)

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

            # plot progress
            self._assess_losses()
            is_final_step = (istep == n_epochs-1)
            if is_final_step or istep%self.plt_conf['n_skip'] == 0:
                self.make_plots(istep)

        # write a bash script that can be run to make gifs
        self._gif()

    def make_plots(self, istep):

        # prepare
        n_epochs = self.trn_conf['n_epochs']
        is_final_step = (istep == n_epochs-1)

        # get classifier predictions on the datasets
        test_pred = self.env.clf_predict(self._test)
        train_pred = self.env.clf_predict(self._train)
        ss_pred = self.env.clf_predict(self._ss)

        # and construct score objects
        test_name = '{}_test_{}'.format(self.clf.name, istep)
        clf_test_score = self.assess_clf(test_name, test_pred, self._test['Y'], self._test['W'], ss_pred)
        clf_test_score.save(os.path.join(self.score_loc, clf_test_score.fname))

        train_name = '{}_train_{}'.format(self.clf.name, istep)
        clf_train_score = self.assess_clf(train_name, train_pred, self._train['Y'], self._train['W'], ss_pred)
        clf_train_score.save(os.path.join(self.score_loc, clf_train_score.fname))

        # plot setup
        bcm_test_plot = {'score':self.bcm_test_score,
                         'label':'{} ({})'.format(self.bcm_conf['type'], 'test'),
                         'colour':defs.dark_blue,
                         'style':'-'}
        bcm_train_plot = {'score':self.bcm_train_score,
                          'label':'{} ({})'.format(self.bcm_conf['type'], 'train'),
                          'colour':defs.dark_blue,
                          'style':':'}
        clf_test_plot = {'score':clf_test_score,
                         'label':'{} ({})'.format(self.clf.name, 'test'),
                         'colour':defs.blue,
                         'style':'-'}
        clf_train_plot = {'score':clf_train_score,
                          'label':'{} ({})'.format(self.clf.name, 'train'),
                          'colour':defs.blue,
                          'style':':'}

        # determine the unique id and location of the plot
        unique_id = 'final' if is_final_step else '{:04d}'.format(istep)
        def loc(name):
            return self.loc if is_final_step else utils.makedir(os.path.join(self.loc, name))

        # loss plot
        plot.losses(self._losses, loc('losses'), unique_id, self.trn_conf, self.plt_conf)

        # roc plot
        #plot.roc_curve(clf_scores, loc('roc_curve'), unique_id, **kwargs)
        plot.roc_curve([bcm_test_plot, bcm_train_plot, clf_test_plot, clf_train_plot], loc('roc_curve'), unique_id)

        # clf output plot
        plot.clf_output([bcm_test_plot, clf_test_plot], loc('clf_output'), unique_id)

        # mass distro plots
        for perc in self.percentiles:
            pname = 'mass_shape_{}p'.format(perc)
            plot.mass_shape([bcm_test_plot, clf_test_plot], perc, loc(pname), unique_id)

    def assess_clf(self, name, test_pred, test_label, test_weight, ss_pred):

        # roc curves
        roc = self._get_roc(test_pred, test_label, test_weight)

        # clf distros for different mass ranges
        clf_hists = self._get_clf_hists(ss_pred)

        # mass distros for different clf percentiles
        mass_hists = self._get_mass_hists(ss_pred)

        # return the score object
        return ClassifierScore(name, roc, clf_hists, mass_hists)

    def _assess_losses(self):

        C, A, CA = self.env.losses(self._test)

        self._losses['C'].append(C)
        self._losses['A'].append(A)
        self._losses['CA'].append(CA)

    def _get_roc(self, test_pred, test_label, test_weight):

        fprs, tprs, _ = sklearn.metrics.roc_curve(test_label, test_pred.ravel(), sample_weight=test_weight)
        roc_auc = sklearn.metrics.roc_auc_score(test_label, test_pred.ravel(), sample_weight=test_weight)

        return fprs, tprs, roc_auc

    def _get_clf_hists(self, ss_pred):

        # determine the indices of events in a mass range
        mass = self.pre['Z'].inverse_transform(self._ss['Z'])
        ind = {}
        ind['low'] = mass < 120
        ind['high'] = mass > 130
        ind['medium'] = np.logical_and(np.logical_not(ind['low']), np.logical_not(ind['high']))

        # now compute the classifier distribution
        clf_hists = {}
        for mass_range in ind:
            clf_score = ss_pred[ind[mass_range]]
            weights = self._ss['W'][ind[mass_range]]
            clf_hists[mass_range], _ = np.histogram(clf_score, weights=weights, bins=defs.bins, range=(0, 1), density=True)

        return clf_hists

    def _get_mass_hists(self, ss_pred):

        # get real mass values
        mass = self.pre['Z'].inverse_transform(self._ss['Z'])

        mass_hists = {}
        for perc in self.percentiles:

            # determine the mass values in top percentile
            cut = np.percentile(ss_pred, 100-perc)
            sel_mass = mass[ss_pred > cut]
            sel_weights = self._ss['W'][ss_pred > cut]

            # and compute histograms
            sel_hist, _ = np.histogram(sel_mass, weights=sel_weights, bins=defs.bins, range=(defs.mlow, defs.mhigh))
            full_hist, _ = np.histogram(mass, weights=self._ss['W'], bins=defs.bins, range=(defs.mlow, defs.mhigh))

            # pack up in dict
            mass_hists[perc] = sel_hist, full_hist

        return mass_hists

    def _gif(self):

        gifs = ['roc_curve', 'clf_output', 'losses']
        gifs += ['mass_shape_{}p'.format(p) for p in self.percentiles]

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

    def __str__(self):

        return '{}: ROC AUC=0.5, fairness=0.5'.format(self.name)






