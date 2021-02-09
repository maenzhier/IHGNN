# coding = utf8

import os
import pickle

CACHE_OVERWRITE = "overwrite"
CACHE_READ = "read"

cache_base_dir = "./utils/Cache_dir/"


class CacheFile(object):
    def __init__(self, path, cache_mod, read_func):
        self.file_path = path
        self.cache_mod = cache_mod
        self.read_func = read_func

        # cache rule is path basename with .pkl
        self.cache_path = "{}{}.pkl".format(cache_base_dir, os.path.basename(self.file_path))

    def build(self):
        if self.cache_mod == CACHE_READ:
            if os.path.exists(self.cache_path):

                with open(self.cache_path, "rb") as f:
                    cache_file = pickle.load(f)
            else:
                raise Exception("file without cache : {}".format(self.file_path))
        elif self.cache_mod == CACHE_OVERWRITE:
            cache_file = self.read_func(self.file_path)

            with open(self.cache_path, "wb") as f:
                pickle.dump(cache_file, f)
        else:
            raise Exception("cache_mod wrong: {}".format(self.cache_mod))

        return cache_file


class CacheVar(object):
    def __init__(self):
        pass

    def build(self, var_name, var):
        cache_path = "{}{}.pkl".format(cache_base_dir, var_name)
        with open(cache_path, "wb") as f:
            pickle.dump(var, f)

    def get_cache_by_name(self, var_name):
        cache_path = "{}{}.pkl".format(cache_base_dir, var_name)
        if os.path.exists(cache_path):

            with open(cache_path, "rb") as f:
                cache_var = pickle.load(f)
        else:
            raise Exception("file without cache : {}".format(self.file_path))

        return cache_var