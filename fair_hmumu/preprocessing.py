from sklearn.decomposition import PCA
from fair_hmumu.utils import Saveable


class PCAWhiteningPreprocessor(Saveable):

    def __init__(self, n_cpts):

        super().__init__('PCA')
        self.pca = PCA(n_cpts, svd_solver='auto', whiten=True)

    def fit(self, train_data):

        print('--- Fitting PCA')

        self.pca.fit(train_data)

    def transform(self, data):

        return self.pca.transform(data)

    def inverse_transform(self, data):

        return self.pca.inverse_transform(data)

