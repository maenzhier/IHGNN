# coding=utf8

import random
from hgnn.HetG import HetG
from collections import Counter


def het_walk_restart(hetG: HetG, max_total_neighbor_n, max_sampling_neighbor_nums: dict):
    print("starting to sample Heterogeneous graph by random walk with restart")
    neighbors = []
    sampled_neighbors = []
    for node_type in hetG.node_types:
        neighbors.append(hetG.neighbors[node_type])
        node_n = hetG.node_types_n_dict[node_type]
        sampled_neighbors.append([[] for k in range(node_n)])

    sampling_neighbor_nums = {}
    #print(max_sampling_neighbor_nums)

    for i in range(len(hetG.node_types)):
        node_type = hetG.node_types[i]
        node_n = hetG.node_types_n_dict[node_type]
        for id in range(node_n):
            start_node = node_type + str(id)
            curNode = node_type + str(id)
            sampled_neighbor_list = sampled_neighbors[i][id]

            if len(neighbors[i][id]) > 0:
                neigh_len = 0
                for t in hetG.node_types:
                    sampling_neighbor_nums[t] = 0

                while neigh_len < max_total_neighbor_n:  # maximum neighbor size = 100
                    rand_p = random.random()  # return p
                    if rand_p > 0.5:
                        cur_node_type = curNode[0]
                        type_id = hetG.node_types.index(cur_node_type)
                        node_id = int(curNode[1:])
                        neighbor_list = neighbors[type_id][node_id]

                        nextNode = random.choice(neighbor_list)

                        next_node_type = nextNode[0]
                        if nextNode != start_node \
                                and sampling_neighbor_nums[next_node_type] < max_sampling_neighbor_nums[next_node_type]:
                            sampled_neighbor_list.append(nextNode)
                            sampling_neighbor_nums[next_node_type] += 1
                            neigh_len += 1

                        curNode = nextNode
                    else:
                        curNode = node_type + str(id)

                #if len(sampled_neighbor_list) < max_total_neighbor_n:
                #    print(sampled_neighbor_list)

            print("start_node : {} ".format(start_node))
            if len(sampled_neighbor_list) < max_total_neighbor_n:
                print(sampled_neighbor_list)

    sampled_neighbors_dict = {}
    for i in range(len(hetG.node_types)):
        node_type = hetG.node_types[i]
        node_n = hetG.node_types_n_dict[node_type]
        for j in range(node_n):
            sampled_neighbors[i][j] = list(sampled_neighbors[i][j])
        sampled_neighbors_dict[node_type] = sampled_neighbors[i]

    return sampled_neighbors, sampled_neighbors_dict


def gen_type_based_sampled_neighbors(hetG: HetG, sampled_neighbors):
    print("starting to generate type_based sampled neighbors")
    type_based_sampled_neighbors = []
    for node_type in hetG.node_types:
        node_n = hetG.node_types_n_dict[node_type]
        temp = [[[] for k in range(node_n)] for m in range(len(hetG.node_types))]
        type_based_sampled_neighbors.append(temp)

    for i in range(len(hetG.node_types)):
        node_type = hetG.node_types[i]
        sampled_neighbor_list = sampled_neighbors[i]
        node_n = hetG.node_types_n_dict[node_type]
        type_based_sampled_neighbor_list = type_based_sampled_neighbors[i]
        for j in range(node_n):
            neighbor_temp = sampled_neighbor_list[j]
            for nei in neighbor_temp:
                nei_type = nei[0]
                type_id = hetG.node_types.index(nei_type)
                type_based_sampled_neighbor_list[type_id][j].append(nei)

    return type_based_sampled_neighbors


def gen_topN_sampled_neighbors(hetG: HetG, type_based_sampled_neighbors, topN_sampled_neighbors_nums):
    topN_sampled_neighbors = []
    for node_type in hetG.node_types:
        node_n = hetG.node_types_n_dict[node_type]
        temp = [[[] for k in range(node_n)] for m in range(len(hetG.node_types))]
        topN_sampled_neighbors.append(temp)

    for i in range(len(hetG.node_types)):
        node_type = hetG.node_types[i]
        type_based_sampled_neighbor_list = type_based_sampled_neighbors[i]
        node_n = hetG.node_types_n_dict[node_type]
        topN_sampled_neighbor_list = topN_sampled_neighbors[i]
        for j in range(node_n):
            for k in range(len(hetG.node_types)):
                topN_size = topN_sampled_neighbors_nums[hetG.node_types[k]]
                neighbor_temp = Counter(type_based_sampled_neighbor_list[k][j])
                topN_list = neighbor_temp.most_common(topN_size)

                for m in range(len(topN_list)):
                    topN_sampled_neighbor_list[k][j].append(topN_list[m][0])

                if len(topN_list) and len(topN_list) < topN_size:
                    for n in range(len(topN_list), topN_size):
                        topN_sampled_neighbor_list[k][j].\
                            append(random.choice(topN_sampled_neighbor_list[k][j]))

    return topN_sampled_neighbors



