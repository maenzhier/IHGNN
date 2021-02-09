# coding = utf8

import tensorflow as tf
from tensorflow.python import keras

import math
import numpy as np

from tqdm import tqdm
import pickle

tf.enable_eager_execution()


class Embedding(keras.Model):

    def __init__(self, vocab_size, embedding_size, pretrained_embeddings=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding_size = embedding_size

        if pretrained_embeddings is None:
            self.embeddings = tf.Variable(
                tf.truncated_normal([vocab_size, embedding_size], dtype=tf.float32,
                                    stddev=1.0 / math.sqrt(embedding_size)))
        else:
            self.embeddings = tf.Variable(pretrained_embeddings, dtype=tf.float32, trainable=False)

        # self.dropout_model = keras.layers.Dropout(5e-1)

        self.pic_encoder = keras.Sequential([
            keras.layers.Dense(embedding_size, activation=tf.nn.relu),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(self.embedding_size)
        ])

    def call(self, inputs, training=None, mask=None):
        pic_embeddings = tf.nn.embedding_lookup(self.embeddings, inputs)
        return self.pic_encoder(pic_embeddings, training=training)

    def get_pic_embeddings(self):
        return self.pic_encoder(self.embeddings, training=False)


class CSR(keras.Model):

    def __init__(self, vocab_size, attr_matrix, pic_embedding_model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attr_matrix = attr_matrix

        self.vocab_size = vocab_size

        self.pic_embedding_model = pic_embedding_model

        self.attr_embeddings = tf.constant(self.attr_matrix, dtype=tf.float32)
        self.user_encoder = keras.Sequential([
            keras.layers.Dense(pic_embedding_model.embedding_size)
        ])

    def call(self, inputs, training=None, mask=None):
        user_ids, photo_index = inputs
        user_attrs = tf.nn.embedding_lookup(self.attr_embeddings, user_ids)

        encoded_users = self.user_encoder(user_attrs, training=training)

        encoded_pics = self.encode_photos(photo_index, training=training)
        logits = tf.reduce_sum(encoded_users * encoded_pics, axis=1)

        return logits

    def encode_photos(self, photo_index, training):
        embedded_pics = self.pic_embedding_model(photo_index, training=training)
        return embedded_pics

    def encode_all_users(self):
        return self.user_encoder(self.attr_embeddings, training=False)

    def encode_all_photos(self, training=False):
        return self.pic_embedding_model.get_pic_embeddings()


def gcn_layer(photo_to_users_list, encoded_attrs_vec):
    sampling_num = 150
    m_b = []

    for users_indices in photo_to_users_list:
        users_indices = np.asarray(users_indices)
        if len(users_indices) > sampling_num:
            users_indices = np.random.choice(users_indices, sampling_num, replace=False)

        m_b.append(np.sum(encoded_attrs_vec[users_indices], axis=0))

    return np.asarray(m_b)


def gcn_photo_by_user(photo_to_users_list, encoded_attrs_vec):

    photo_embeddings = gcn_layer(photo_to_users_list, encoded_attrs_vec)
    #attr_matrix = gcn_layer(interactive_matrix, photo_embeddings)
    #photo_embeddings = gcn_layer(interactive_matrix.T, attr_matrix)

    return photo_embeddings


def gen_pos_pairs(train_user_photos_list, batch_size):
    filtered_user_indices = []
    filtered_pic_indices = []
    for i, photos in enumerate(train_user_photos_list):
        for photo_index in photos:
            filtered_user_indices.append(i)
            filtered_pic_indices.append(photo_index)

    for user_indices, pos_pic_indices in tf.data.Dataset.from_tensor_slices((filtered_user_indices, filtered_pic_indices)).shuffle(batch_size * 5)\
        .batch(batch_size).prefetch(5):
        yield user_indices, pos_pic_indices


def gen_triples(train_user_photos_list, vocab_size, batch_size, num_neg_samples=1):
    max_neg = vocab_size
    for user_indices, pos_pic_indices in gen_pos_pairs(train_user_photos_list, batch_size):
        neg_pic_indices_list = [np.random.randint(0, max_neg, (pos_pic_indices.shape[0]))
                                for _ in range(num_neg_samples)]
        yield user_indices, pos_pic_indices, [tf.convert_to_tensor(neg_pic_indices) for neg_pic_indices in neg_pic_indices_list]


def train(interaction_and_encoded_attr_mat_info):

    photo_list = interaction_and_encoded_attr_mat_info["photo_list"]
    photo_id_to_index = interaction_and_encoded_attr_mat_info["photo_id_to_index"]
    photo_to_users_list = interaction_and_encoded_attr_mat_info["photo_to_users_list"]
    user_list = interaction_and_encoded_attr_mat_info["user_list"]
    user_id_to_index = interaction_and_encoded_attr_mat_info["user_id_to_index"]
    user_to_photos_list = interaction_and_encoded_attr_mat_info["user_to_photos_list"]
    encoded_attrs_vec = interaction_and_encoded_attr_mat_info["attrs_vec"]

    split_rate = 0.8
    split = int(len(user_list) * split_rate)
    all_user_indices = np.arange(len(user_list))
    random_indices = np.random.permutation(all_user_indices)
    train_indices = random_indices[:split]
    test_indices = random_indices[split:]

    train_user_photos_list = [user_to_photos_list[index] for index in train_indices]
    vocab_size = len(photo_list)

    photo_gcn_embeddings = gcn_photo_by_user(photo_to_users_list, encoded_attrs_vec)

    embedding_size = 300

    pic_embedding_model = Embedding(vocab_size, embedding_size, photo_gcn_embeddings)
    model = CSR(vocab_size, encoded_attrs_vec, pic_embedding_model)

    learning_rate = 1e-1
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    adam_optimizer = tf.train.AdamOptimizer(2e-3)

    for epoch in range(0, 2000):
        print("\nepoch: ", epoch)

        for step, batch in tqdm(enumerate(gen_triples(train_user_photos_list, vocab_size, 4000, 1))):
            user_indices, pos_pic_indices, neg_pic_indices_list = batch

            with tf.GradientTape() as tape:
                pos_logits = model([user_indices, pos_pic_indices], training=True)

                pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(pos_logits),
                    logits=pos_logits
                )

                neg_losses = []
                for neg_pic_indices in neg_pic_indices_list:
                    neg_logits = model([user_indices, neg_pic_indices], training=True)

                    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.zeros_like(neg_logits),
                        logits=neg_logits
                    )
                    neg_losses.append(neg_loss)

                losses = pos_losses + tf.add_n(neg_losses)
                mean_loss = tf.reduce_mean(losses)

                kernel_vars = [var for var in tape.watched_variables() if "kernel" in var.name]
                l2_losses = [tf.nn.l2_loss(var) for var in kernel_vars]
                l2_loss = tf.add_n(l2_losses)

                mean_loss += l2_loss * 5e-1

            vars = tape.watched_variables()
            grads = tape.gradient(mean_loss, vars)
            # optimizer.apply_gradients(zip(grads, vars))

            dense_vars_and_grads = [gv for gv in zip(grads, vars) if "dense" in gv[1].name]
            #embedding_vars_and_grads = [gv for gv in zip(grads, vars) if "dense" not in gv[1].name]
            #optimizer.apply_gradients(embedding_vars_and_grads)
            adam_optimizer.apply_gradients(dense_vars_and_grads)

        if epoch % 10 == 9:
            pic_mean_loss = None
            print("epoch: {}, loss: {}, pic_loss: {}".format(epoch, mean_loss, pic_mean_loss))
            encoded_users = np.asarray(model.encode_all_users())
            test_encoded_users = encoded_users[test_indices]

            prob = tf.sigmoid(test_encoded_users @ tf.transpose(model.encode_all_photos(), [1, 0]))

            test_user_photos_gt = [user_to_photos_list[index] for index in test_indices]

            with open("output.p", "wb") as f:
                pickle.dump([test_user_photos_gt, prob.numpy()], f)
