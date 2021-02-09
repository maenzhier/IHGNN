# coding=utf8

import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *


class HetG(object):
    def __init__(self, node_types_n_tuples):
        super(HetG, self).__init__()

        self.node_types = []
        self.node_types_n_dict = {}

        for tup in node_types_n_tuples:
            t = tup[0]
            num = tup[1]
            self.node_types.append(t)
            self.node_types_n_dict[t] = num

        self.relation_types = []
        self.relations = {}

        self.neighbor_types = {}
        self.neighbors = {}

        self.node_embeddings = {}

    def print_obj(self):
        print(self.__dict__)

    def load_relation(self, src_node_type, tar_node_type, relation_f_path):
        src_node_num = self.node_types_n_dict[src_node_type]
        src_tar_list = [[] for k in range(src_node_num)]

        rel_type = tuple([src_node_type, tar_node_type])
        print("relation: {} loading ... ".format(rel_type))

        rel_f = open(relation_f_path, "r")
        for line in rel_f:
            line = line.strip()
            node_id = int(re.split(':', line)[0])
            neigh_list = re.split(':', line)[1]
            neigh_list_id = re.split(',', neigh_list)

            for neigh_id in neigh_list_id:
                src_tar_list[node_id].append(tar_node_type + str(neigh_id))
        rel_f.close()

        self.relation_types.append(rel_type)
        self.relations[rel_type] = src_tar_list

    def gen_neighbors(self):
        for t in self.node_types:
            self.neighbor_types[t] = []

        for rel_t in self.relation_types:
            src_node_type = rel_t[0]
            tar_node_type = rel_t[1]
            self.neighbor_types[src_node_type].append(tar_node_type)

        for t in self.node_types:
            neighbor_type_list = self.neighbor_types[t]

            if len(neighbor_type_list) == 1:
                rel_type = tuple([t, neighbor_type_list[0]])
                self.neighbors[t] = self.relations[rel_type]
            elif len(neighbor_type_list) > 1:
                relations = []
                for neighbor_type in neighbor_type_list:
                    rel_type = tuple([t, neighbor_type])
                    relations.append(self.relations[rel_type])

                t_num = self.node_types_n_dict[t]
                neighbor_list = [[] for k in range(t_num)]
                for i in range(t_num):
                    for rel in relations:
                        neighbor_list[i] += rel[i]

                self.neighbors[t] = neighbor_list

    def load_embeddings(self, node_type, embedding_f_path, embed_d):
        embedding_f = open(embedding_f_path, "r")

        node_n = self.node_types_n_dict[node_type]

        embeddings = np.zeros((node_n, embed_d))
        for line in islice(embedding_f, 0, None):
            values = line.split()
            index = int(values[0])
            embeds = np.asarray(values[1:], dtype='float32')
            embeddings[index] = embeds

        embedding_f.close()

        self.node_embeddings[node_type] = embeddings
