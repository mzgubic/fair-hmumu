import os
import uproot as ur
import pandas as pd


class DatasetHandler:
    """
    Class which handles datasets and splitting into training and test sets.
    """

    def __init__(self, production, features, nentries=1000):

        # settings
        self.production = production
        self.loc = os.path.join(os.getenv('DATA'), production)
        self.features = features
        self.nentries = nentries
        self.datasets = ['sig', 'data', 'ss']
        
        # set up
        self.setup()

    def setup(self):

        self.df = {}
        for dataset in self.datasets:
            #self.df[dataset] = 
            pass


        

production = '20190116_default'
features = ['Muons_PT_Lead', 'Muons_PT_Sub']
dh = DatasetHandler(production, features)
