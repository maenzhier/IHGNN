# coding = utf8
import pickle
import pandas as pd


class Feature(object):
    def __init__(self, features):
        self.features = features

        self.features_to_id_dict = dict(zip(features, range(0, len(features))))

        self.features_val_num = len(self.features)

        self.wv = {}

    def build_embeddings(self):
        for feature in self.features:
            vec = [0.] * self.features_val_num
            vec[self.features_to_id_dict[feature]] = 1.
            self.wv[feature] = vec

    def save_embeddings(self, pkl_path):
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)


def test():
    # feature_name = "city_name"
    # feature_name = "province_name"
    # feature_name = "mod_brand"
    feature_name = "mod"

    df = pd.read_csv("../data/10000_user_photo_click_info.csv", sep=" ")
    df.fillna("", inplace=True)
    df[feature_name] = df[feature_name].map(lambda fea: fea.strip(","))
    feas = df[feature_name].unique()
    apps = feas.tolist()
    print(apps)

    path = "../embeddings/other_feature/10000_user_{}_one_hot_vec.pkl".format(feature_name)
    feature = Feature(apps)
    feature.build_embeddings()
    feature.save_embeddings(path)

    #
    # with open(path, "rb") as f:
    #     app_list = pickle.load(f)
    #
    # print(app_list.wv["com.tencent.LEGOCUBE"])

# test()