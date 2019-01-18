import os
import sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import fair_hmumu.defs as defs
import fair_hmumu.plot as plot
import fair_hmumu.models as models
import fair_hmumu.utils as utils
from fair_hmumu.dataset import DatasetHandler
from fair_hmumu.preprocessing import PCAWhiteningPreprocessor
from fair_hmumu.environment import TFEnvironment


class Trainer:

    def __init__(self, run_conf):

        # configurations
        self.loc = run_conf.loc
        self.clf = models.Classifier('DNN', run_conf.get('Classifier'))
        self.adv = None #TODO
        self.opt_conf = run_conf.get('Optimiser')
        self.trn_conf = run_conf.get('Training')
        self.bcm_conf = run_conf.get('Benchmark')

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
        self._fit_preprocessing()

        # set up TensorFlow environment
        self._setup_environment()

    def _load_data(self):
 
        # data handling
        production = self.trn_conf['production']
        features = ['Z_PT', 'Muons_CosThetaStar']
        entrystop = self.trn_conf['entrystop']
        self.dh = DatasetHandler(production, features, entrystop=entrystop, test_frac=0.25, seed=42)

        # load 
        self._train = self.dh.get_train(defs.jet0) # TODO: jet channels
        self._test = self.dh.get_test(defs.jet0) # TODO: jet channels
        self._ss = self.dh.get_ss(defs.jet0) # TODO: jet channels

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

    def _train_benchmarks(self):

        print('--- Training the benchmark {} model'.format(self.bcm_conf['type']))

        # construct hyperparameters
        bcm_hps = dict(self.bcm_conf)
        bcm_hps.pop('type')

        # instantiate the model
        if self.bcm_conf['type'] == 'GBC':
            self.bcm = GradientBoostingClassifier(**bcm_hps)

        # train the model
        self.bcm.fit(self._train['X'], self._train['Y'].ravel(), sample_weight=self._train['W'].ravel())

        print('--- Making benchmark prediction on the test and ss events')

        # predict on the test set
        test_pred = self.bcm.predict_proba(self._test['X'])[:, 1].reshape(-1, 1)
        ss_pred = self.bcm.predict_proba(self._ss['X'])[:, 1].reshape(-1, 1)

        # and store the score
        self.bcm_score = self.assess_clf(self.bcm_conf['type'], test_pred, ss_pred) 
        self.bcm_score.save(os.path.join(self.loc, self.bcm_score.fname))

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

        print('--- Pretraining for {} steps'.format(self.trn_conf['n_pre']))

        # pretrain the classifier
        for _ in range(self.trn_conf['n_pre']):
            batch = self.dh.get_batch(defs.jet0)
            batch = self.transform(batch)
            self.env.pretrain_step(batch)

        # pretrain the adversary
        for _ in range(self.trn_conf['n_pre']):
            batch = self.dh.get_batch(defs.jet0)
            batch = self.transform(batch)
            self.env.train_step_adv(batch)

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

            # only plot every ten steps and the final one
            is_final_step = (istep == n_epochs-1)
            if is_final_step or istep%10 == 0:

                # asses classifier performance
                test_pred = None
                ss_pred = None 
                #clf_score = self.assess_clf('{}_{}'.format(self.clf.name, istep), test_pred, ss_pred) 
                #clf_score.save(os.path.join(self.loc, clf_score.fname))

                # plot setup 
                clf_scores = [self.bcm_score]
                labels = [self.bcm_conf['type']]
                colours = ['k']
                styles = ['-']
                unique_id = 'final' if is_final_step else str(istep)

                # make plots
                #loc = self.loc if is_final_step else utils.makedir(os.path.join(self.loc, 'roc_curve'))
                #plot.roc_curve(clf_scores, labels, colours, styles, loc, unique_id)

                loc = self.loc if is_final_step else utils.makedir(os.path.join(self.loc, 'clf_output'))
                plot.clf_output(clf_scores, labels, colours, styles, loc, unique_id)


    def assess_clf(self, name, test_pred, ss_pred):

        ############
        # roc curves
        ############

        fprs, tprs, _ = sklearn.metrics.roc_curve(self._test['Y'], test_pred.ravel(), sample_weight=self._test['W'])
        roc_auc = sklearn.metrics.roc_auc_score(self._test['Y'], test_pred.ravel(), sample_weight=self._test['W'])
        roc = fprs, tprs, roc_auc

        ############
        # clf distros for different mass ranges
        ############

        mass = self.pre['Z'].inverse_transform(self._ss['Z'])

        # determine the indices of events in a mass range
        ind = {}
        ind['low'] = mass < 120
        ind['high'] = mass > 130
        ind['medium'] = np.logical_and(np.logical_not(ind['low']), np.logical_not(ind['high']))

        # now compute the classifier distribution
        clf_score, clf_hist = {}, {}
        for mass_range in ind:
            clf_score[mass_range] = ss_pred[ind[mass_range]]
            clf_hist[mass_range], _ = np.histogram(clf_score[mass_range], bins=defs.bins, range=(0, 1))

        ############
        # fairness
        ############

        # construct the score object
        score = ClassifierScore(name, roc, clf_hist)

        return score


class ClassifierScore(utils.Saveable):

    def __init__(self, name, roc, clf_hist):

        self.name = name
        self.fname = '{}_score.pkl'.format(self.name)
        self.roc_curve = roc[:2]
        self.roc_auc = roc[2]
        self.clf_hist = clf_hist

    def __str__(self):

        return '{}: ROC AUC=0.5, fairness=0.5'.format(self.name)









