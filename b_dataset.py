# coding = utf8

import numpy as np
from functools import reduce

from utils.cache import CacheFile, CacheVar
from utils.embedding import Embedding
from autoencoder import AutoEncoder


class Dataset(object):
    def __init__(self, params_maps):
        self.params_map = params_maps

        self.raw_data_info = self.params_map["raw_data_info"]

        self.df = None
        self.photo_list = None
        self.photo_id_to_index = {}
        self.photo_to_users_list = None
        self.user_list = None
        self.user_id_to_index = {}
        self.user_to_photos_list = None

        self.attr_columns = None
        self.attrs_df = None

        self.interactive_columns = ["user_id", "photo_id"]
        self.united_attr_col_new_name = "united_attr_col"

        self.attr_embedding_map = self.params_map["attr_embedding_map"]
        self.loaded_attr_embedding_map = {}

    def read_data(self):
        print("reading data ....")
        path = self.raw_data_info["path"]
        cache_mod = self.raw_data_info["cache_mod"]
        read_func = self.raw_data_info["read_func"]

        cache = CacheFile(path, cache_mod, read_func)
        self.df = cache.build()

    def check_data(self):
        print("checking data ....")
        self.df.fillna("", inplace=True)
        assert self.df[self.df["is_click"] == 0].shape[0] == 0

    def filter_by_columns(self):
        print("filter by columns ....")
        self.attr_columns = [k for k in self.attr_embedding_map]
        self.df = self.df[self.interactive_columns + self.attr_columns]

    def gather_user_photo_info(self):

        self.user_list = self.df["user_id"].unique()
        self.user_id_to_index = dict(zip(self.user_list, range(0, len(self.user_list))))

        self.photo_list = self.df["photo_id"].unique()
        self.photo_id_to_index = dict(zip(self.photo_list, range(0, len(self.photo_list))))

        self.df["user_id"] = self.df["user_id"].map(lambda x: self.user_id_to_index[x])
        self.df["photo_id"] = self.df["photo_id"].map(lambda x: self.photo_id_to_index[x])

        interactive_matrix_df = self.df[self.interactive_columns]
        agg_dict = {
            "photo_id": lambda x: x.unique(),
            "user_id": lambda x: list(x.unique())
        }
        grouped_by_photo = interactive_matrix_df.groupby("photo_id").agg(agg_dict)
        grouped_by_photo = grouped_by_photo.reset_index(drop=True)
        grouped_by_photo.sort_values(by="photo_id")
        self.photo_to_users_list = grouped_by_photo["user_id"].tolist()

    def group_by_column(self, column_name="user_id"):
        def handle_app_list(x):
            # print(x)
            x = x.map(lambda apps: set(apps.strip(",").split(",")))
            y = x.tolist()
            y = reduce(lambda a, b: a | b, y)
            return list(y)

        agg_dict = {}
        for name in self.df.columns.values:
            if name == column_name:
                agg_dict[name] = lambda x: x.unique()
            elif name == "app_list":
                agg_dict["app_list"] = handle_app_list
            else:
                agg_dict[name] = lambda x: list(x.unique())

        grouped = self.df.groupby(column_name)[self.df.columns.values].agg(agg_dict)
        grouped = grouped.reset_index(drop=True)
        grouped.sort_values(by="user_id")
        self.user_to_photos_list = grouped["photo_id"].tolist()

        self.attrs_df = grouped[self.attr_columns]
        self.df = None

    # embeddings using KeyedVectors format
    def load_attr_embeddings(self):
        for attr_name in self.attr_embedding_map.keys():
            print("load attr embedding for : {}".format(attr_name))

            attr_embedding_info = self.attr_embedding_map[attr_name]
            path = attr_embedding_info["path"]
            cache_mod = attr_embedding_info["cache_mod"]
            read_func = attr_embedding_info["read_func"]

            attr_embedding = Embedding(path, cache_mod, read_func)
            attr_embedding.load()

            self.loaded_attr_embedding_map[attr_name] = attr_embedding

    def calulate_embeddings_for_column(self, column_name, pretrained_embeddings):

        def cal_embedings(values):
            assert type(values) == list
            embeddings = []
            for value in values:
                embeddings.append(pretrained_embeddings.word_vec(value))

            embeddings = np.asarray(embeddings)
            return np.sum(embeddings, axis=0)

        self.attrs_df[column_name] = self.attrs_df[column_name].map(cal_embedings)

    def build_embeded_attr_columns(self):
        print("start to build embeded attr column ...")
        for column_name in self.attrs_df.columns.values:
            print("cal embedding for col : {}".format(column_name))
            pretrained_embeddings = self.loaded_attr_embedding_map[column_name]
            self.calulate_embeddings_for_column(column_name, pretrained_embeddings)
        print("end to build embeded attr column ...")

    def gen_attr_column(self):
        print("gen new attr column with {}".format(self.united_attr_col_new_name))

        def concat(x):
            y = x.tolist()
            y = reduce(lambda a, b: np.append(a, b), y)
            return list(y)

        self.df[self.united_attr_col_new_name] = self.df[self.attr_columns].apply(concat, axis=1)
        self.interaction_and_attribute_df = self.df[self.interactive_columns + [self.united_attr_col_new_name]]

    def get_attrs_vec(self):
        print("gen new attr column with {}".format(self.united_attr_col_new_name))

        def concat(x):
            y = x.tolist()
            y = reduce(lambda a, b: np.append(a, b), y)
            return list(y)

        attrs_vec = np.asarray(self.attrs_df.apply(concat, axis=1).tolist())
        self.attrs_df = None
        return attrs_vec


    # with cache funciton
    def build_interaction_and_attribute_df(self, interaction_and_attribute_info_name):
        print("start to build interaction and attribute df .... ")

        self.read_data()
        self.check_data()
        self.filter_by_columns()
        self.gather_user_photo_info()
        self.group_by_column(column_name="user_id")
        self.load_attr_embeddings()
        self.build_embeded_attr_columns()

        attrs_vec = self.get_attrs_vec()

        print("end to build interaction and attribute info.... ")
        interaction_and_attribute_info = {
            "photo_list": self.photo_list,
            "photo_id_to_index": self.photo_id_to_index,
            "photo_to_users_list": self.photo_to_users_list,
            "user_list": self.user_list,
            "user_id_to_index": self.user_id_to_index,
            "user_to_photos_list": self.user_to_photos_list,
            "attrs_vec": attrs_vec
        }

        cache_var = CacheVar()
        cache_var.build(interaction_and_attribute_info_name, interaction_and_attribute_info)

    def read_interaction_and_attribute_info(self, interaction_and_attribute_info_name):

        cache_var = CacheVar()
        interaction_and_attribute_info = cache_var.get_cache_by_name(interaction_and_attribute_info_name)

        return interaction_and_attribute_info

    # with cache function
    def build_interaction_and_encoded_attr_mat(self, interaction_and_attribute_info, attr_encoded_dim,
                                               interaction_and_encoded_attr_mat_info_name):

        print("start to build interaction and encode attribute matrix .... ")

        attrs_vec = interaction_and_attribute_info["attrs_vec"]
        attr_dim = len(attrs_vec[0])
        print("encode attr embedding from {} to {}".format(attr_dim, attr_encoded_dim))

        print(attrs_vec.shape)
        autoEncoder = AutoEncoder(attrs_vec, target_dim=attr_encoded_dim)
        autoEncoder.train()

        print("end to build interaction and encode attribute matrix .... ")

        self.encoded_attr_vec = np.asarray(autoEncoder.get_coded_val())
        print("encoded_attr_vec : {}".format(self.encoded_attr_vec.shape))
        print(self.encoded_attr_vec)

        interaction_and_attribute_info["attrs_vec"] = self.encoded_attr_vec

        cache_var = CacheVar()
        cache_var.build(interaction_and_encoded_attr_mat_info_name,  interaction_and_attribute_info)

    def read_interaction_and_encoded_attr_mat(self, interaction_and_encoded_attr_mat_info_name):

        cache_var = CacheVar()
        interaction_and_encoded_attr_mat_info = cache_var.get_cache_by_name(interaction_and_encoded_attr_mat_info_name)
        return interaction_and_encoded_attr_mat_info


class PhotoDataSet(object):
    def __init__(self, params):
        self.raw_data_info = params["raw_data_info"]

        self.df = None

        self.photo_id_column_name = "photo_id"
        self.tag_name_column = "tag_names"
        self.video_feature_column = "list_to_str(feature)"

        self.category_list = None
        self.category_dict = None

        self.united_photo_feature_name = "photo_feature"

    def read_data(self):
        print("reading photos data ....")
        path = self.raw_data_info["path"]
        cache_mod = self.raw_data_info["cache_mod"]
        read_func = self.raw_data_info["read_func"]

        cache = CacheFile(path, cache_mod, read_func)
        self.df = cache.build()

    def check_data(self):
        print("checking data ....")
        self.df.fillna("-1", inplace=True)

    def convert_str_list(self, column_name=None):
        print("convert string column to list : {}".format(column_name))
        if column_name != None:
            self.df[column_name] = self.df[column_name].map(lambda x: x.strip(",").strip("_").split("_"))
        else:
            raise Exception("convert column {} :str to list wrong!".format(column_name))

    def convert_column_to_one_hot_vec(self, column_name=None):
        if column_name not in self.df.columns.values:
            raise Exception("convert column {} :str to list wrong!".format(column_name))
        print("build one hot vec for column : {}".format(column_name))

        category_list = self.df[column_name].tolist()
        category_list = reduce(lambda a, b: np.append(a, b), category_list)
        self.category_list = list(set(category_list))

        self.category_dict = dict(zip(category_list, range(0, len(category_list))))

        def handle_tags(tags):
            tag_indics = [self.category_dict[tag] for tag in tags]
            tag_vec = np.asarray([0.] * len(category_list))
            tag_vec[tag_indics] = 1.
            return tag_vec

        self.df[column_name] = self.df[column_name].map(handle_tags)

    def gen_photo_vecs(self):
        print("gen united column for photo feature ....")

        def concat(x):
            x = [np.asarray(w).astype(np.float) for w in x.tolist()]
            x = reduce(lambda a, b: np.append(a, b), x)
            return list(x)

        # self.df[self.united_photo_feature_name] = \
        #     self.df[[self.tag_name_column, self.video_feature_column]].apply(concat, axis=1)

        self.df[self.united_photo_feature_name] = \
            self.df[[self.video_feature_column]].apply(concat, axis=1)

        self.df = self.df[[self.photo_id_column_name, self.united_photo_feature_name]]

    def reset_photo_indices(self, interactive_photo_list):
        # print(self.df.shape)
        interactive_photo_list = interactive_photo_list.tolist()
        # df = self.df[self.df[self.photo_id_column_name] in interactive_photo_list]
        # assert df.shape[0] == len(interactive_photo_list)
        self.df["photo_index"] = \
            self.df[self.photo_id_column_name].map(lambda photo_id: interactive_photo_list.index(photo_id))
        self.df = self.df.sort_values(by="photo_index")
        self.df = self.df[[self.united_photo_feature_name]]

    def build_photo_vec_mat(self, interactive_photo_list, photo_vec_mat_name):

        self.read_data()
        self.check_data()
        # self.convert_str_list(self.tag_name_column)
        self.convert_str_list(self.video_feature_column)
        # self.convert_column_to_one_hot_vec(self.tag_name_column)
        self.gen_photo_vecs()
        self.reset_photo_indices(interactive_photo_list)

        photo_vec_mat = np.asarray(self.df[self.united_photo_feature_name].tolist())
        #
        # autoEncoder = AutoEncoder(photo_vec_mat,
        #                           target_dim=500, lr=0.001, epochs=50, other_hidden_dim=1024)
        # autoEncoder.train()
        #
        # photo_vec_mat = autoEncoder.get_coded_val()

        cache_var = CacheVar()
        cache_var.build(photo_vec_mat_name, photo_vec_mat)

    def read_iphoto_vec_mat(self, photo_vec_mat_name):

        cache_var = CacheVar()
        photo_vec_mat = cache_var.get_cache_by_name(photo_vec_mat_name)
        return photo_vec_mat





