import os
import fair_hmumu.defs as defs
import fair_hmumu.models as models
from fair_hmumu.dataset import DatasetHandler
from fair_hmumu.preprocessing import PCAWhiteningPreprocessor


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

        self.train = self.dh.get_train(defs.jet0) # TODO: jet channels

        # preprocessing
        self.pre = PCAWhiteningPreprocessor(n_cpts=self.train['X'].shape[1])
        self.pre_nuis = PCAWhiteningPreprocessor(n_cpts=self.train['Z'].shape[1])
        self.pre.fit(self.train['X'])
        self.pre_nuis.fit(self.train['Z'])
        self.pre.save(os.path.join(self.loc, 'X_{}.pkl'.format(defs.jet0)))
        self.pre_nuis.save(os.path.join(self.loc, 'Z_{}.pkl'.format(defs.jet0)))

        # environment





