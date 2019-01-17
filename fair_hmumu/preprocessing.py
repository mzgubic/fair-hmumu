import pickle
import numpy as np
from sklearn.decomposition import PCA
from fair_hmumu.utils import Saveable


class PCAWhiteningPreprocessor(Saveable):

    def __init__(self, n_cpts):

        self.pca = PCA(n_cpts, svd_solver='auto', whiten=True)

    def fit(self, train_data):
        
        self.pca.fit(train_data)

    def process(self, data):

        return self.pca.transform(data)
       
