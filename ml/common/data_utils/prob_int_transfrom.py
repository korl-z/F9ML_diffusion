import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ProbabilityIntegralTransform(BaseEstimator, TransformerMixin):
    """
    Empirical Probability Integral Transform (ECDF) with inverse.
    Transforms each column to approximately Uniform(eps, 1-eps).
    """
    def __init__(self, eps=1e-6):
        self.eps = float(eps)
        self.x_vals_ = None
        self.cdf_vals_ = None
        self.n_samples_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, d = X.shape
        self.n_samples_ = n
        self.x_vals_ = []
        self.cdf_vals_ = []
        for j in range(d):
            x = X[:, j]
            sorted_x = np.sort(x)
            uniq, idx_start, counts = np.unique(sorted_x, return_index=True, return_counts=True)
            # average index for duplicates (0-based)
            avg_idx = idx_start + (counts - 1) / 2.0
            # convert to CDF values in (0,1) using (i+0.5)/n rule
            cdf = (avg_idx + 0.5) / n
            cdf = np.clip(cdf, self.eps, 1.0 - self.eps)
            self.x_vals_.append(uniq)
            self.cdf_vals_.append(cdf)
        return self

    def transform(self, X):
        if self.x_vals_ is None or self.cdf_vals_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        X = np.asarray(X)
        single_dim = False
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            single_dim = True
        n, d = X.shape
        if d != len(self.x_vals_):
            raise ValueError("Number of features in X does not match fit data.")
        U = np.empty_like(X, dtype=float)
        for j in range(d):
            U[:, j] = np.interp(
                X[:, j],
                self.x_vals_[j],
                self.cdf_vals_[j],
                left=self.cdf_vals_[j][0],
                right=self.cdf_vals_[j][-1],
            )
        U = (U - 0.5) * np.sqrt(12.0)
        return U.ravel() if single_dim else U

    def inverse_transform(self, U):
        if self.x_vals_ is None or self.cdf_vals_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        U = np.asarray(U)
        single_dim = False
        if U.ndim == 1:
            U = U.reshape(-1, 1)
            single_dim = True
        n, d = U.shape
        if d != len(self.x_vals_):
            raise ValueError("Number of features in U does not match fit data.")
        X = np.empty_like(U, dtype=float)
        U = U / np.sqrt(12.0) + 0.5
        for j in range(d):
            X[:, j] = np.interp(
                U[:, j],
                self.cdf_vals_[j],
                self.x_vals_[j],
                left=self.x_vals_[j][0],
                right=self.x_vals_[j][-1],
            )
        return X.ravel() if single_dim else X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)