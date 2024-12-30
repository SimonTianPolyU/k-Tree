
from PkTree import PkTree
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class PkRF(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, dt_params=None, k=None, max_features=10, bootstrap=True):
        self.n_estimators = n_estimators
        self.dt_params = dt_params
        self.k = k
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        for _ in range(self.n_estimators):
            if self.bootstrap:
                X_sample, y_sample = resample(X, y)
            else:
                X_sample, y_sample = X, y

            # Random feature selection
            max_features = self.max_features
            if max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif max_features == 'log2':
                max_features = int(np.log2(n_features))
            elif isinstance(max_features, float):
                max_features = int(max_features * n_features)

            features = np.random.choice(range(n_features), max_features, replace=False)
            X_sample = X_sample[:, features]

            # Create and train a PkTree
            tree = PkTree(self.dt_params, self.k)
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, features))

        return self

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        effects = 0
        for tree, features in self.trees:
            preds += tree.predict(X[:, features])[0]  # Ensure to use the first output for prediction
            effects += tree.predict(X[:, features])[1]
        return preds / self.n_estimators, effects / self.n_estimators

