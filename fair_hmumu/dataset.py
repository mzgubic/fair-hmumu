import os
import itertools
import numpy as np
import uproot as ur
import pandas as pd
from fair_hmumu import defs


class DatasetHandler:

    def __init__(self, production, njet, features, entrystop=1000, test_frac=0.25, seed=42):
        """
        Dataset handler. Split in training and test sets, get training batches.

        Args:
            production (str): Name of the input dataset production
            features (list of str): List of branch names to be used in the training.
            entrystop (int or None): Maximum number of read rows. None reads all.
        """

        # settings
        self.production = production
        self.njet = njet
        self.loc = os.path.join(os.getenv('DATA'), production)
        self.features = self._list_features(features)
        self.branches = self.features + [defs.target, defs.mass, defs.weight]
        self.entrystop = entrystop
        self.seed = seed
        self.test_frac = test_frac

        # set up
        self._load()
        self._split()

    def _load(self):

        print('--- Loading the datasets')

        self.df = {ds:{} for ds in defs.datasets}

        for dataset in defs.datasets:

            # get the tree
            fname = '{}/{}.root'.format(self.loc, dataset)
            tree = ur.open(fname)[self.njet]

            # get the array
            arrays = tree.arrays(branches=self.branches, entrystop=self.entrystop)

            # convert it to a DataFrame and decode column names to strings
            self.df[dataset]['full'] = pd.DataFrame(arrays)
            self.df[dataset]['full'].columns = [feature.decode('utf-8') for feature in self.df[dataset]['full'].columns]

    def _split(self):

        print('--- Splitting into training and test sets')

        for dataset in defs.datasets:

            # get the training and test set indices
            nentries = self.df[dataset]['full'].shape[0]
            indices = np.arange(nentries)
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            split_at = int(self.test_frac*nentries)
            ind_train, ind_test = indices[split_at:], indices[:split_at]

            # split
            self.df[dataset]['train'] = self.df[dataset]['full'].iloc[ind_train]
            self.df[dataset]['test'] = self.df[dataset]['full'].iloc[ind_test]

    def _xyzw(self, df):

        X = df[self.features].values
        Y = df[defs.target].values.reshape(-1, 1)
        Z = df[defs.mass].values.reshape(-1, 1)
        W = df[defs.weight].values.reshape(-1, 1)

        return {'X':X, 'Y':Y, 'Z':Z, 'W':W}

    def _list_features(self, features):

        # TODO: add dependence on njets 
        # features preparation
        muon_vectors_pt = ['Muons_Eta_Lead', 'Muons_Eta_Sub', 'Muons_Phi_Lead', 'Muons_Phi_Sub', 'Muons_PT_Lead', 'Muons_PT_Sub']
        muon_vectors_ptm = ['Muons_Eta_Lead', 'Muons_Eta_Sub', 'Muons_Phi_Lead', 'Muons_Phi_Sub', 'Muons_PTM_Lead', 'Muons_PTM_Sub']
        dimuon_vector_pt = ['Z_Eta', 'Z_Phi', 'Z_PT']
        dimuon_vector_ptm = ['Z_Eta', 'Z_Phi', 'Z_PTM']

        flist = []
        if features == 'pt/mass':
            flist = muon_vectors_ptm + dimuon_vector_ptm

        return flist
        

    def get_train(self):
        """
        Get the entire training set.
        """
        print('--- Fetching full training set')

        # fetch components
        sig = self.df[defs.sig]['train']
        data = self.df[defs.data]['train']

        # concatenate and reshuffle
        result = pd.concat([sig, data])
        result = result.sample(frac=1).reset_index(drop=True)

        print('-> {} events'.format(result.shape[0]))

        return self._xyzw(result)

    def get_test(self):
        """
        Get the entire test set.
        """
        print('--- Fetching full test set')

        # fetch components
        sig = self.df[defs.sig]['test']
        data = self.df[defs.data]['test']

        # concatenate and reshuffle
        result = pd.concat([sig, data])
        result = result.sample(frac=1).reset_index(drop=True)

        print('-> {} events'.format(result.shape[0]))

        return self._xyzw(result)

    def get_ss(self, nentries=None):
        """
        Get nentries events from spurious signal data.
        """

        # fetch full ds
        df = self.df[defs.ss]['full']

        # num entries in df
        all_entries = df.shape[0]

        # how many to fetch
        if nentries is None or nentries < 0:
            nentries = all_entries
        else:
            nentries = min(nentries, all_entries)

        print('--- Fetching {}/{} spurious signal events.'.format(nentries, all_entries))

        return self._xyzw(df.iloc[:nentries])

    def get_batch(self, batchsize=512):
        """
        Get a balanced batch of randomly sampled training data.
        """
        assert batchsize % 2 == 0, "batchsize must be even"

        # fetch signal and background events
        sig = self.df[defs.sig]['train']
        data = self.df[defs.data]['train']

        # sample half batchsize from each
        neach = int(batchsize/2.)
        sig_inds = np.random.randint(sig.shape[0], size=neach)
        data_inds = np.random.randint(data.shape[0], size=neach)

        sig = sig.iloc[sig_inds].copy()
        data = data.iloc[data_inds].copy()

        # normalise weights
        sig.loc[:, defs.weight] = sig.loc[:, defs.weight] * (neach / np.sum(sig.loc[:, defs.weight]))
        data.loc[:, defs.weight] = data.loc[:, defs.weight] * (neach / np.sum(data.loc[:, defs.weight]))

        # concatenate and reshuffle
        result = pd.concat([sig, data])
        result = result.sample(frac=1).reset_index(drop=True)

        return self._xyzw(result)


