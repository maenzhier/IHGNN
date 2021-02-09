# coding=utf8

import tensorflow as tf
import numpy as np

from functools import reduce


class AutoEncoder(object):
    def __init__(self, data, attr_emb_map=None, target_dim=30, lr=0.0015, batch_size=600, epochs=100,
                 other_hidden_dim=500):
        self.data = data
        self.target_dim = target_dim

        self.attr_emb_map = attr_emb_map

        self.input_dim = 0
        for col_name in self.attr_emb_map:
            pretrained_embed = self.attr_emb_map[col_name]
            self.input_dim += pretrained_embed.embedding_dim

        self.learning_rate = lr
        self.batch_size = batch_size
        self.epoch_num = epochs

        self.other_hidden_dim = other_hidden_dim

        self.coding_val = None

        self.corrputed_data = None

    def corrupting_data(self, input):
        corrputed_input = input + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=input.shape)

        num = int(0.7 * np.prod(input.shape))

        random_row = np.random.randint(input.shape[0], size=num)
        random_col = np.random.randint(input.shape[1], size=num)

        corrputed_input[random_row, random_col] = 0.0

        return corrputed_input

    def create_raw_batch_data(self):
        raw_batchs = [self.data[x:x + self.batch_size] for x in range(0, self.data.shape[0], self.batch_size)]
        for raw_batch in raw_batchs:
            yield raw_batch

    def calulate_embeddings_for_column(self, df, column_name, pretrained_embeddings):

        def cal_embedings(values):
            assert type(values) == list
            embeddings = []
            for value in values:
                embeddings.append(pretrained_embeddings.word_vec(value))

            embeddings = np.asarray(embeddings)
            return np.sum(embeddings, axis=0)

        df[column_name] = df[column_name].map(cal_embedings)
        return df

    def build_attrs_embeddings(self, raw_batch):
        for col_name in raw_batch.columns.values:
            pretrained_embeddings = self.attr_emb_map[col_name]
            raw_batch = \
                self.calulate_embeddings_for_column(raw_batch, col_name, pretrained_embeddings)

        def concat(x):
            y = x.tolist()
            y = reduce(lambda a, b: np.append(a, b), y)
            return list(y)

        raw_batch = raw_batch.apply(concat, axis=1)
        raw_batch = np.asarray(raw_batch.tolist())

        return raw_batch

    def create_batch_data(self):
        for raw_batch in self.create_raw_batch_data():
            batch = self.build_attrs_embeddings(raw_batch)
            yield batch

    def encoder(self, X):
        hidden_0 = tf.layers.dense(X, self.other_hidden_dim, activation=tf.nn.sigmoid)
        target_hidden = tf.layers.dense(hidden_0, self.target_dim, activity_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        return target_hidden

    def train(self):

        X = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        corrputed_X = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        target_hidden = self.encoder(corrputed_X)
        hidden_1 = tf.layers.dense(target_hidden, self.other_hidden_dim, activation=tf.nn.sigmoid)
        output = tf.layers.dense(hidden_1, self.input_dim)

        l2_loss = tf.losses.get_regularization_loss()
        #loss = tf.reduce_mean(tf.square(output - X))

        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(X, output)) + 0.4 * l2_loss

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        training_opt = optimizer.minimize(loss)

        init = tf.global_variables_initializer

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.epoch_num):
                for batch in self.create_batch_data():
                    # print(batch.shape)
                    Loss, _ = sess.run([loss, training_opt], feed_dict={X: batch,
                                                                        corrputed_X: self.corrupting_data(batch)})
                if epoch % 50 >= 0:
                    print("epoch: {}, loss: {}".format(epoch, Loss))

            self.coding_val = target_hidden.eval(feed_dict={X: self.data,
                                                            corrputed_X: self.data})

    def get_coded_val(self):
        return self.coding_val

