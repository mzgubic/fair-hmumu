import os
from sklearn.ensemble import GradientBoostingClassifier
import fair_hmumu.defs as defs
import fair_hmumu.models as models
from fair_hmumu.dataset import DatasetHandler
from fair_hmumu.preprocessing import PCAWhiteningPreprocessor
from fair_hmumu.environment import TFEnvironment


class Trainer:

    def __init__(self, run_conf):

        # configurations
        self.loc = run_conf.loc
        self.clf = models.Classifier('clf', run_conf.get('Classifier'))
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

        # data handling
        production = self.trn_conf['production']
        high_level = ['Z_PT', 'Muons_CosThetaStar']
        entrystop = self.trn_conf['entrystop']
        self.dh = DatasetHandler(production, high_level, entrystop=entrystop, test_frac=0.25, seed=42)

        self._train = self.dh.get_train(defs.jet0) # TODO: jet channels
        self._test = self.dh.get_test(defs.jet0) # TODO: jet channels
        batch_example = self.dh.get_batch(defs.jet0)

        # preprocessing
        self.pre = PCAWhiteningPreprocessor(n_cpts=self._train['X'].shape[1])
        self.pre_nuis = PCAWhiteningPreprocessor(n_cpts=self._train['Z'].shape[1])
        self.pre.fit(self._train['X'])
        self.pre_nuis.fit(self._train['Z'])
        self.pre.save(os.path.join(self.loc, 'PCA_X_{}.pkl'.format(defs.jet0)))
        self.pre_nuis.save(os.path.join(self.loc, 'PCA_Z_{}.pkl'.format(defs.jet0)))

        # benchmark training
        self.train_benchmarks()

        # environment
        self.env = TFEnvironment(self.clf, self.adv, self.opt_conf)
        self.env.build(batch_example)
        self.env.initialise_variables()

    def train_benchmarks(self):

        print('--- Training the benchmark {} model'.format(self.bcm_conf['type']))

        # construct hyperparameters
        bcm_hps = dict(self.bcm_conf)
        bcm_hps.pop('type')

        # instantiate the model
        if self.bcm_conf['type'] == 'GBC':
            self.bcm = GradientBoostingClassifier(**bcm_hps)

        # train the model
        self.bcm.fit(self._train['X'], self._train['Y'].ravel(), sample_weight=self._train['W'].ravel())

        # predict on the test set
        self.bcm.predict_proba(self._test['X'])[:, 1]

        # and store 
        # TODO: make a score object?

    def pretrain(self):

        print('--- Pretraining for {} steps'.format(self.trn_conf['n_pre']))

        # pretrain the classifier
        for _ in range(self.trn_conf['n_pre']):
            batch = self.dh.get_batch(defs.jet0)
            self.env.pretrain_step(batch)

        # pretrain the adversary
        for _ in range(self.trn_conf['n_pre']):
            batch = self.dh.get_batch(defs.jet0)
            self.env.train_step_adv(batch)

    def train(self):

        print('--- Training for {} steps'.format(self.trn_conf['n_epochs']))
    
        # training step: clf and adv
        for istep in range(self.trn_conf['n_epochs']):
            
            # train the classifier
            for _ in range(self.trn_conf['n_clf']):
                batch = self.dh.get_batch(defs.jet0)
                self.env.train_step_clf(batch)

            # train the adversary
            for _ in range(self.trn_conf['n_adv']):
                batch = self.dh.get_batch(defs.jet0)
                self.env.train_step_adv(batch)

            # monitor performance
            # TODO
        




