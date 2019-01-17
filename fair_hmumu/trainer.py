import os
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

        print('------------')
        print('--- Settings:')
        print(self.clf)
        print(self.adv)
        print(self.opt_conf)
        print(self.trn_conf)
        print('------------')

        # data handling
        production = self.trn_conf['production']
        high_level = ['Z_PT', 'Muons_CosThetaStar']
        entrystop = self.trn_conf['entrystop']
        self.dh = DatasetHandler(production, high_level, entrystop=entrystop, test_frac=0.25, seed=42)

        self._train = self.dh.get_train(defs.jet0) # TODO: jet channels
        batch_example = self.dh.get_batch(defs.jet0)

        # preprocessing
        self.pre = PCAWhiteningPreprocessor(n_cpts=self._train['X'].shape[1])
        self.pre_nuis = PCAWhiteningPreprocessor(n_cpts=self._train['Z'].shape[1])
        self.pre.fit(self._train['X'])
        self.pre_nuis.fit(self._train['Z'])
        self.pre.save(os.path.join(self.loc, 'PCA_X_{}.pkl'.format(defs.jet0)))
        self.pre_nuis.save(os.path.join(self.loc, 'PCA_Z_{}.pkl'.format(defs.jet0)))

        # environment
        self.env = TFEnvironment(self.clf, self.adv, self.opt_conf)
        self.env.build(batch_example)
        self.env.initialise_variables()

    def pretrain(self):

        print('--- Training for {} steps'.format(self.trn_conf['n_pre']))

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
    
        # train and plot progress
        for istep in range(self.trn_conf['n_epochs']):
            
            # train the classifier
            for _ in range(self.trn_conf['n_clf']):
                batch = self.dh.get_batch(defs.jet0)
                self.env.train_step_clf(batch)

            # train the adversary
            for _ in range(self.trn_conf['n_adv']):
                batch = self.dh.get_batch(defs.jet0)
                self.env.train_step_adv(batch)
                
            
        




