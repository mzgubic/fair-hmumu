import fair_hmumu.models as models
from fair_hmumu.dataset import DatasetHandler


class Trainer:

    def __init__(self, clf_conf, adv_conf, opt_conf, trn_conf):

        # configurations
        self.clf = models.Classifier('clf', clf_conf)
        self.adv = None #TODO
        self.opt_conf = opt_conf
        self.trn_conf = trn_conf

        print(self.clf)
        print(adv_conf)
        print(opt_conf)
        print(trn_conf)

        # data handling
        production = self.trn_conf['production']
        high_level = ['Z_PT', 'Muons_CosThetaStar']
        entrystop = self.trn_conf['entrystop']
        self.dh = DatasetHandler(production, high_level, entrystop=entrystop, test_frac=0.25, seed=42)

        # preprocessing



