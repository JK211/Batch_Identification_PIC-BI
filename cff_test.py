#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-09-19
@LastEditor: JK211
LastEditTime: 2024-04-15
@Discription: This .py is a standalone implementation of d Cover-free Family Identification (d-CFF(t,n)) .
                Here we demonstrate the use of 2-CFF(25,125) for batch identification.
@Environment: python 3.7
'''
import numpy as np
import math
np.set_printoptions(suppress=True)

Nt = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


lt = 0  # the total number of identification tests until this round
vt = 0  # the number of identification tests required for this round of testing
Mt = []  # set of requests that passed the identification in this round.
Ct = []  # set of illegal requests identified in this round.

new_Nt = []
Split_Group = []
n = 125
m = 25
d = 2

# n = 27
# m = 9
# d = 1
#  First group Nt, the group size is 125
group_size = n - 1
group_nums = len(Nt) / n
ceil_group_nums = math.ceil(group_nums)
for i in range(ceil_group_nums):
    if i < int(group_nums):
        temp = Nt[(group_size * i + i): (group_size * i + i) + group_size + 1]
        Split_Group.append(temp)
    else:
        temp = Nt[(group_size * i + i): (len(Nt) - 1) + 1]
        Split_Group.append(temp)
if len(Split_Group[ceil_group_nums - 1]) > n / 2:
    # If the number of the last group is less than n/2, fill it up to n
    temp = [0] * (n - len(Split_Group[ceil_group_nums - 1]))
    Split_Group[ceil_group_nums - 1] = np.append(Split_Group[ceil_group_nums - 1], temp)
#  After grouping, execute 2-CFF(25, 125) on each group in turn. If a group is not completed, skip and add Nt and process it in the next round.
index = 0
# M = np.load("./Cover free families demos/1-cff9-27.npy")
M = np.load("./Cover free families demos/2-cff25-125.npy")
print("Split_Group大小为：", len(Split_Group))
print("Split Group为：", Split_Group)

# while len(Split_Group[index]) == n:

all_dist_keys = []
for item in Split_Group:
    if len(item) == n:
        # print("+++++++++++++++")
        print("item为：", item)
        valid_set = []
        invalid_set = []
        indist_set = []
        count = 0
        cff_batch = []
        new_Nt_keys = []
        vt = vt + m
        for i in range(len(M)):
            # Here they are grouped by 25, each group is 125, and marked with serial numbers to facilitate subsequent processing.
            temp = {}
            for j in range(len(M[i])):
                if M[i][j] == 1:
                    temp[j] = item[j]
            cff_batch.append(temp)
        # print("-------------", cff_batch)

        all_keys = [x for x in range(0, len(M[0]), 1)]
        count = np.sum(item)
        for subset in cff_batch:
            subset_keys = list(subset.keys())
            subset_values = list(subset.values())

            if np.sum(subset_values) == 0:
                valid_set = list(set(valid_set).union(subset_keys))
        Mt = np.append(Mt, valid_set)

        if count <= d:
            invalid_set = list(set(all_keys).difference(valid_set))
            Ct = np.append(Ct, invalid_set)
        else:
            indist_set = list(set(all_keys).difference(valid_set))
            all_dist_keys = np.append(all_dist_keys, indist_set)
            new_Nt_keys = np.append(new_Nt_keys, indist_set)
            # indist_set_values = [x for x in subset[indist_set]]

            indist_set_values = []
            for x in indist_set:
                indist_set_values.append(item[x])
            new_Nt = np.append(new_Nt, indist_set_values)

if len(Split_Group[len(Split_Group) - 1]) < n:
    new_Nt = np.append(new_Nt, Split_Group[len(Split_Group) - 1])
lt = lt + vt

print("The legal signature set Mt is:", Mt)
print("The number of legal signatures is:", len(Mt))
print("The illegal signature set Ct is:", Ct)
print("The number of illegal signatures is:", len(Ct))
print("new_Nt_keys为：", new_Nt_keys)
print("new_Nt为：", list(new_Nt))
print(len(new_Nt))
print("all_dist_keys为：", all_dist_keys)