import os
import itertools
import numpy as np
import uproot as ur
import pandas as pd


class DatasetHandler:

    def __init__(self, production, features, entrystop=1000, test_frac=0.25, seed=42):
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
        self.seed = seed
        self.test_frac = test_frac
        self.datasets = ['sig', 'data', 'ss']
        self.njets = ['0jet', '1jet', '2jet']
        
        # set up
        self._load()
        self._split()

    def _load(self):

        print('--- Loading the datasets')

        self.df = {ds:{njet:{} for njet in self.njets} for ds in self.datasets}

        for dataset, njet in itertools.product(self.datasets, self.njets):

            # get the tree
            fname = '{}/{}.root'.format(self.loc, dataset)
            tree = ur.open(fname)[njet]

            # get the array
            arrays = tree.arrays(branches=self.features, entrystop=self.entrystop)

            # convert it to a DataFrame and decode column names to strings
            self.df[dataset][njet]['full'] = pd.DataFrame(arrays)
            self.df[dataset][njet]['full'].columns = [feature.decode('utf-8') for feature in self.df[dataset][njet]['full'].columns]

    def _split(self):

        print('--- Splitting into training and test sets')

        for dataset, njet in itertools.product(self.datasets, self.njets):

            # get the training and test set indices
            nentries = self.df[dataset][njet]['full'].shape[0]
            indices = np.arange(nentries)
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            split_at = int(self.test_frac*nentries)
            ind_train, ind_test = indices[split_at:], indices[:split_at]

            # split
            self.df[dataset][njet]['train'] = self.df[dataset][njet]['full'].iloc[ind_train]
            self.df[dataset][njet]['test'] = self.df[dataset][njet]['full'].iloc[ind_test]



production = '20190116_default'
features = ['Muons_PT_Lead', 'Muons_PT_Sub']
entrystop=10
dh = DatasetHandler(production, features, entrystop=entrystop)
