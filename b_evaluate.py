# coding=utf-8

import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import pickle

with open("output.p", "rb") as f:
    test_user_photos_gt, prob = pickle.load(f)

# prob = (prob * 100).astype(np.int) / 100.0


def sampling_test_data(test_user_photos_gt, prob):
    photo_num = len(prob[0])
    gt = []
    scores = []
    for user_photos, user_photo_prob in zip(test_user_photos_gt, prob):
        pos_gt_list = [1.] * len(user_photos)
        pos_prob_list = user_photo_prob[np.asarray(user_photos)]

        mask = np.ones(photo_num, dtype=np.bool)
        mask[np.asarray(user_photos)] = False

        all_neg_photo_prob = user_photo_prob[mask]
        neg_num = len(pos_gt_list) * 10
        neg_prob_list = np.random.choice(all_neg_photo_prob, neg_num, replace=False)

        gt.extend(list(pos_gt_list) + [0.0] * neg_num)
        scores.extend(list(pos_prob_list) + list(neg_prob_list))

    return gt, scores

gt, scores = sampling_test_data(test_user_photos_gt, prob)
print(gt[:100])
print(scores[:100])

auc_score, auc_op = tf.metrics.auc(np.asarray(gt), np.asarray(scores))
#auc_score, auc_op = tf.metrics.auc(test_interactive_matrix_gt, prob)
# auc_score, auc_op = tf.metrics.auc(test_interactive_matrix_gt[0::20], prob[0::20])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run(tf.local_variables_initializer())
    print(sess.run([auc_score, auc_op]))


# print(test_interactive_matrix_gt, prob)