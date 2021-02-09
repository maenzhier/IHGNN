# coding = utf8
from utils.cache import CacheFile
import numpy as np
from feature.app_list import AppList


class Embedding(object):
    def __init__(self, path, cache_mod, read_func):
        self.embedding_file_path = path
        self.cache_mod = cache_mod
        self.read_func = read_func

        self.embedding = None

        self.embedding_dim = None

        self.not_in_vocab_words_embeddings = {}

    def load(self):
        cache = CacheFile(self.embedding_file_path, self.cache_mod, self.read_func)
        self.embedding = cache.build()
        if type(self.embedding) == AppList:
            self.embedding_dim = self.embedding.app_num
        else:
            self.embedding_dim = 300

    def word_vec(self, word):
        try:
            embedding = self.embedding.wv[word]
        except:
            if word in self.not_in_vocab_words_embeddings:
                embedding = self.not_in_vocab_words_embeddings[word]
            else:
                self.not_in_vocab_words_embeddings[word] = np.random.randn(self.embedding_dim)
                embedding = self.not_in_vocab_words_embeddings[word]

        return embedding
