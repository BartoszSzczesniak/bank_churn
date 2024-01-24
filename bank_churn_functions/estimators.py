from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from types import FunctionType

class FeatureBuilder(BaseEstimator, TransformerMixin):

    def __init__(self, config: dict[str, FunctionType]) -> None:
        """---
        Tool for building new features.

        ## Parameters
        config: dictionary defining new features. Keys must contain new feature names and values must contain a functions to build the new features."""
        
        super().__init__()

        self.config = config
        self.feature_names_out_ = list(self.config.keys())

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        X = X.assign(**self.config)
        X = X.loc[:, self.feature_names_out_]
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_