from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
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

class OutputTransformer(Saveable):

    def __init__(self, n_quantiles=1000, output_distribution='uniform'):

        super().__init__('QuantileTransformer')
        self.tsf = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution)

    def fit(self, clf_output):

        print('--- Fitting classifier output transformer')

        self.tsf.fit(clf_output)

    def transform(self, data):

        return self.tsf.transform(data)

    def inverse_transform(self, data):

        return self.tsf.inverse_transform(data)
        
