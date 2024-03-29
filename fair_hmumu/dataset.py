import os
import itertools
import numpy as np
import uproot as ur
import pandas as pd
from fair_hmumu import defs
from fair_hmumu.utils import timeit


class DatasetHandler:

    def __init__(self, trn_conf, seed=42):
        """
        Dataset handler. Split in training and test sets, get training batches.

        Args:
            production (str): Name of the input dataset production
            features (list of str): List of all features
            entrystop (int or None): Maximum number of read rows. None reads all.
        """

        # general settings
        self.trn_conf = trn_conf
        self.production = trn_conf['production']
        self.loc = os.path.join(os.getenv('DATA'), self.production)
        self.njet = trn_conf['njet']
        self.entrystop = trn_conf['entrystop']
        self.n_div = trn_conf['n_div']
        self.n_rmd = trn_conf['n_rmd']
        self.seed = seed

        # features
        self.bcm_features = feature_list(trn_conf['bcm_features'], self.njet)
        self.clf_features = feature_list(trn_conf['clf_features'], self.njet)
        self.features = list(set(self.bcm_features) | set(self.clf_features))
        self.branches = self.features + [defs.target, defs.mass, defs.weight, defs.event_number]

        # set up
        print('--- Selecting features for {} channel'.format(self.njet))
        print('-> {}'.format(self.features))
        self._load()
        self._split()

    @timeit
    def _load(self):

        print('--- Loading the datasets')

        self.df = {ds:{} for ds in defs.datasets}

        for dataset in defs.datasets:

            # get the tree
            fname = '{}/{}.root'.format(self.loc, dataset)
            tree = ur.open(fname)[self.njet]

            # get the arrayand convert it to a dataframe
            arrays = tree.arrays(branches=self.branches, entrystop=self.entrystop)
            df = pd.DataFrame(arrays)

            # decode column names to strings
            df.columns = [feature.decode('utf-8') for feature in df.columns]

            # and cast uint to int
            df[defs.event_number] = df[defs.event_number].astype(int)

            # finally save as an attribute
            self.df[dataset]['full'] = df


    @timeit
    def _split(self):

        print('--- Splitting into training and test sets')

        for dataset in defs.datasets:

            # get the training and test set indices, based on event number
            test_condition = '{} % {} == {}'.format(defs.event_number, self.n_div, self.n_rmd)
            is_test = self.df[dataset]['full'].eval(test_condition)
            ind_test = is_test.loc[is_test == True].index
            ind_train = is_test.loc[is_test == False].index

            # split
            self.df[dataset]['train'] = self.df[dataset]['full'].iloc[ind_train]
            self.df[dataset]['test'] = self.df[dataset]['full'].iloc[ind_test]

    def _xyzw(self, df, features):

        X = df[features].values
        Y = df[defs.target].values.reshape(-1, 1)
        Z = df[defs.mass].values.reshape(-1, 1)
        W = df[defs.weight].values.reshape(-1, 1)

        return {'X':X, 'Y':Y, 'Z':Z, 'W':W}

    def _augment(self, data):

        # determine how many
        N = int(data.shape[0] * self.trn_conf['augment'])
        sample = data.iloc[np.random.randint(data.shape[0], size=N)]
        rotate_by = np.random.uniform(2*np.pi, size=N)

        for feature in sample.columns.values:

            if 'Phi' not in feature:
                continue

            # rotate the phi variable
            new_phi = sample[feature] + rotate_by

            # make sure it is in the correct range
            # np.where(cond, value) assigns value where cond is False
            new_phi = new_phi.where(new_phi < np.pi, new_phi - 2*np.pi)

            # and replace the old phi values by rotated ones
            sample[feature] = new_phi

        return sample

    @timeit
    def get_train(self, features):
        """
        Get the entire training set.
        """
        print('--- Fetching full training set')

        # fetch components
        sig = self.df[defs.sig]['train']
        data = self.df[defs.data]['train']

        # concatenate and reshuffle
        result = pd.concat([sig, data])
        result = result.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        print('({} events)'.format(result.shape[0]))

        # augment if asked
        try:
            if self.trn_conf['augment'] > 0:
                result = self._augment(result)
        except KeyError:
            pass

        return self._xyzw(result, features)

    @timeit
    def get_test(self, features):
        """
        Get the entire test set.
        """
        print('--- Fetching full test set')

        # fetch components
        sig = self.df[defs.sig]['test']
        data = self.df[defs.data]['test']

        # concatenate and reshuffle
        result = pd.concat([sig, data])
        result = result.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        print('({} events)'.format(result.shape[0]))

        return self._xyzw(result, features)

    @timeit
    def get_ss(self, features, nentries=None):
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

        return self._xyzw(df.iloc[:nentries], features)

    def get_batch(self, features, batchsize=512):
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
        result = result.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        return self._xyzw(result, features)


def feature_list(collections, njet):

    # feature list
    fset = set()

    # muon features preparation
    muon_vectors_pt = ['Muons_Eta_Lead', 'Muons_Eta_Sub', 'Muons_Phi_Lead', 'Muons_Phi_Sub', 'Muons_PT_Lead', 'Muons_PT_Sub']
    muon_vectors_ptm = ['Muons_Eta_Lead', 'Muons_Eta_Sub', 'Muons_Phi_Lead', 'Muons_Phi_Sub', 'Muons_PTM_Lead', 'Muons_PTM_Sub']

    # dimuon
    dimuon_vector_pt = ['Z_Eta', 'Z_Phi', 'Z_PT']
    dimuon_vector_ptm = ['Z_Eta', 'Z_Phi', 'Z_PTM']

    # jet
    lead_jet = ['Jets_PT_Lead', 'Jets_Eta_Lead', 'Jets_Phi_Lead']
    sub_jet = ['Jets_PT_Sub', 'Jets_Eta_Sub', 'Jets_Phi_Sub']

    # dijet
    dijet = ['Jets_PT_jj', 'Jets_Minv_jj']

    # how to add one collection
    def add_collection(collection):
        if collection == 'pt/mass':
            for var in muon_vectors_ptm + dimuon_vector_ptm:
                fset.add(var)

        elif collection == 'pt':
            for var in muon_vectors_pt + dimuon_vector_pt:
                fset.add(var)

        if njet == defs.jet1:
            for var in lead_jet:
                fset.add(var)

        if njet == defs.jet2:
            for var in lead_jet + sub_jet + dijet:
                fset.add(var)
    
    # actually
    if isinstance(collections, str):
        add_collection(collections)

    elif isinstance(collections, list):
        for col in collections:
            add_collection(col)

    return sorted(list(fset))

