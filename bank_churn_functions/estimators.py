import numpy as np
import pandas as pd
import nltk
from types import FunctionType
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin
from gensim import corpora

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

class TopNWordMoversEmbedder(BaseEstimator, TransformerMixin):

    def __init__(self, corpus: corpora.Dictionary, embed_v_len: int = 100, w2v_v_len: int = 100, phrase_len: int = 2) -> None:
        """---
        Tool for Embedding character strings using the TopN Word Movers Distance algorithm.

        Algorithm:
        - calculates Word2Vec embeddings for phrases (ngrams of word's letters)
        - picks top n (embed_v_len) most common names
        - embeds name as a vector of Words Movers Distance values between the name and top n most common names        
        """
        
        super().__init__()

        self.corpus = corpus
        self.embed_v_len = embed_v_len
        self.w2v_v_len = w2v_v_len
        self.phrase_len = phrase_len
        
        self.letter_ngrams = None
        self.top_letter_ngrams = None
        
        self.w2v_model = None
        self.kv = None

    def compute_letter_ngram(self, name):
        "Represents words as ngrams of letters using character-level tokenization approach"

        ngrams = nltk.ngrams(
            sequence=name, 
            n=self.phrase_len, 
            pad_left=True, 
            pad_right=True, 
            left_pad_symbol="^", 
            right_pad_symbol="$"
            )
        
        return tuple(["".join(ngram) for ngram in ngrams])

    def embed_letter_ngram(self, letter_ngram): 
        "Embeds letter ngram representations as vectors of Word Movers Distance between the word and top n most common words"
        return np.array([
            self.w2v_model.wv.wmdistance(top_letter_ngram, letter_ngram)
            for top_letter_ngram in self.top_letter_ngrams
            ])

    def fit(self, X, y=None):

        self.letter_ngrams = [self.compute_letter_ngram(name) for name in list(self.corpus.values())]
        self.top_letter_ngrams = [self.compute_letter_ngram(name) for name, cnt in self.corpus.most_common(self.embed_v_len)]
        
        self.w2v_model = Word2Vec(sentences=self.letter_ngrams, vector_size=self.w2v_v_len, alpha=0.025, window=10, min_count=1)

        embeddings = [self.embed_letter_ngram(letter_ngram) for letter_ngram in self.letter_ngrams]

        top_embedding_kv = KeyedVectors(self.embed_v_len, 0)
        top_embedding_kv.add_vectors(keys=list(self.corpus.values()), weights=embeddings)

        self.kv = top_embedding_kv

        return self
    
    def transform(self, X, y=None):

        return np.stack(np.array([self.kv.get_vector(word) for word in X]), axis=0)

    def get_feature_names_out(self, input_features=None):
        
        if input_features:
            return [f"{input_features[0]}_embd_{i}" for i in range(self.embed_v_len)]
        else:
            return [f"embd_{i}" for i in range(self.embed_v_len)]
        
        