#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-09-19
@LastEditor: JK211
LastEditTime: 2024-01-11
@Discription: This .py is used to model batch requests from sensors. For the sake of simplicity of the experiment, we
set the requests as a list of 0 or 1, where 0 represents a legal request and 1 represents an illegal request. Thus, batch
identification becomes finding out the 1 from the list.
@Environment: python 3.7
'''
import numpy as np
import random as rand


def batch_request_generating(batch_size, invalid_size):
    Sigs2beVefi = np.zeros(batch_size, int)
    for i in range(invalid_size):
        positive_position = rand.randrange(batch_size)

        while Sigs2beVefi[positive_position] == 1:
            positive_position = rand.randrange(batch_size)

        Sigs2beVefi[positive_position] = 1

    return Sigs2beVefi


# Here it is specified to generate a batch request with a concurrency of 300, which is then copied to Batch_Signatures_*actions.py for use.
# 300-10
batch = batch_request_generating(300, 10).tolist()
# 2000-300
# batch = batch_request_generating(2000, 300).tolist()


for j in range(300):
    rand.shuffle(batch)

print(batch)
print(type(batch))