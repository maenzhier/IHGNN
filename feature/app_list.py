# coding = utf8

import pickle
import pandas as pd
import time

from functools import reduce
import gensim


class AppList(object):
    def __init__(self, app_list):
        self.app_list = list(set(app_list))

        self.apps_to_id_dict = dict(zip(app_list, range(0, len(app_list))))

        self.app_num = len(self.app_list)

        self.wv = {}

    # def build_embeddings(self):
    #     wv = self.wv
    #     app_num = self.app_num
    #     apps_to_id_dict = self.apps_to_id_dict
    #
    #     def gen_embedding(app):
    #         print(app)
    #         vec = [0.] * app_num
    #         vec[apps_to_id_dict[app]] = 1.
    #         wv[app] = vec
    #         return app
    #
    #     map(gen_embedding, self.app_list)

    def build_embeddings(self):
        for app in self.app_list:
            vec = [0.] * self.app_num
            vec[self.apps_to_id_dict[app]] = 1.
            self.wv[app] = vec

    def save_embeddings(self, pkl_path):
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)


def test():
    df = pd.read_csv("../data/10000_user_photo_click_info.csv", sep=" ")
    apps = df["app_list"].map(lambda app: set(app.strip(",").split(",")))
    apps = apps.tolist()

    apps = list(reduce(lambda a, b: a | b, apps))
    print(apps)
    print(len(apps))

    path = "../embeddings/apps/10000_user_app_list_one_hot_vec.pkl"
    app_list = AppList(apps)
    app_list.build_embeddings()
    app_list.save_embeddings(path)
    print(app_list.wv)

    # with open(path, "rb") as f:
    #     app_list = pickle.load(f)
    #
    # print(app_list.wv["com.tencent.LEGOCUBE"])

# test()