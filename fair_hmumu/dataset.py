import os
import itertools
import uproot as ur
import pandas as pd


class DatasetHandler:

    def __init__(self, production, features, entrystop=1000):
        """
        Dataset handler. Split in training and test sets, get training batches.

        Args:
            production (str): Name of the input dataset production
            features (list of str): List of branch names to be used in the training.
            entrystop (int or None): Maximum number of read rows. None reads all.
        """

        # settings
        self.production = production
        self.loc = os.path.join(os.getenv('DATA'), production)
        self.features = features
        self.entrystop = entrystop
        self.datasets = ['sig', 'data', 'ss']
        self.njets = ['0jet', '1jet', '2jet']
        
        # set up
        self._setup()

    def _setup(self):

        self.df = {ds:{} for ds in self.datasets}

        print('--- Loading the datasets')
        for dataset, njet in itertools.product(self.datasets, self.njets):

            # get the tree
            fname = '{}/{}.root'.format(self.loc, dataset)
            tree = ur.open(fname)[njet]

            # get the array
            arrays = tree.arrays(branches=self.features, entrystop=self.entrystop)

            # convert it to a DataFrame and decode column names to strings
            self.df[dataset][njet] = pd.DataFrame(arrays)
            self.df[dataset][njet].columns = [feature.decode('utf-8') for feature in self.df[dataset][njet].columns]

        


        

production = '20190116_default'
features = ['Muons_PT_Lead', 'Muons_PT_Sub']
dh = DatasetHandler(production, features)
