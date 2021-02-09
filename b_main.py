# coding = utf8


from utils.cache import CACHE_READ, CACHE_OVERWRITE
from b_dataset import Dataset
from feature.app_list import AppList
from feature.other_feature import Feature

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def read_csv(path):
    import pandas as pd
    import os
    if os.path.exists(path):

        file = pd.read_csv(path, sep=" ")
    else:
        raise Exception("read csv wrong : {}".format(path))
    return file


def read_gensim_word2vec(path):
    import gensim
    import os
    if os.path.exists(path):

        word2vec = gensim.models.KeyedVectors.load_word2vec_format(path)
    else:
        raise Exception("read gensim word2vec wrong : {}".format(path))

    return word2vec


def read_app_list_word2vec(path):
    import os
    import pickle
    print(path)
    if os.path.exists(path):

        with open(path, "rb") as f:
            apps_vec = pickle.load(f)
    else:
        raise Exception("read app list word2vec wrong : {}".format(path))

    return apps_vec


print("start handling interactive matrix and user attr vec ......")

embedding_path = "./embeddings/word-char-ngram/sgns.baidubaike.bigram-char"
attr_encoded_dim = 200

interaction_and_attribute_info_name = "10000_interaction_and_attribute_info"
interaction_and_encoded_attr_mat_info_name = "10000_interaction_and_encoded_attr_mat_info"

app_list_embedding_path = "./embeddings/apps/10000_user_app_list_one_hot_vec.pkl"
raw_data_path = "./data/10000_user_photo_click_info.csv"

raw_data_info = {
    "path": raw_data_path,
    "cache_mod": CACHE_READ,
    "read_func": read_csv
}

attr_embedding_map = {
    "app_list": {"path": app_list_embedding_path, "cache_mod": CACHE_READ, "read_func": read_app_list_word2vec},
    "city_name": {"path": embedding_path, "cache_mod": CACHE_READ, "read_func": read_gensim_word2vec},
    "province_name": {"path": embedding_path, "cache_mod": CACHE_READ, "read_func": read_gensim_word2vec},
    "mod_brand": {"path": embedding_path, "cache_mod": CACHE_READ, "read_func": read_gensim_word2vec},
    "mod": {"path": embedding_path, "cache_mod": CACHE_READ, "read_func": read_gensim_word2vec}
}

params_maps = {
    "raw_data_info": raw_data_info,
    "attr_embedding_map": attr_embedding_map
}

dataset = Dataset(params_maps)

# dataset.build_interaction_and_attribute_df(interaction_and_attribute_info_name)
# interaction_and_attribute_info = dataset.read_interaction_and_attribute_info(interaction_and_attribute_info_name)
#
#
# dataset.build_interaction_and_encoded_attr_mat(interaction_and_attribute_info, attr_encoded_dim,
#                                                interaction_and_encoded_attr_mat_info_name)
interaction_and_encoded_attr_mat_info = dataset.read_interaction_and_encoded_attr_mat(
    interaction_and_encoded_attr_mat_info_name)

# print(interaction_and_encoded_attr_mat_info)

# print("starting cold start reco.....")
#
#
from b_r_cold_start_reco import train
train(interaction_and_encoded_attr_mat_info)

