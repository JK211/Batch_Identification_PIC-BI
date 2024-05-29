#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-09-19
@LastEditor: JK211
LastEditTime: 2024-04-15
@Discription: This .py is used to calculate the cryptographic overhead of the batch authentication we use,
which corresponds to the overhead of the cryptographic primitives in Table 4 of the paper. It is worth noting that the
cost in Table 4 of the paper was obtained on the Raspberry Pi.
@Environment: python 3.7
'''
from tate_bilinear_pairing import eta
from tate_bilinear_pairing import ecc
import random
import time

# eta.init(151)

a = random.randint(0, 1000)
b = random.randint(0, 1000)

g = ecc.gen()


inf1, x1, y1 = ecc.scalar_mult(a, g)
inf2, x2, y2 = ecc.scalar_mult(b, g)

sum_p = 0
for i in range(10):
    t_p_1 = time.perf_counter()
    t = eta.pairing(x1, y1, x2, y2)
    t_p_2 = time.perf_counter()
    sum_p = sum_p + t_p_2 - t_p_1

p1 = [inf2, x2, y2]

sum_mul = 0
for i in range(10):
    k = random.randint(0, 1000)
    t_mul_1 = time.perf_counter()
    p3 = ecc.scalar_mult(k, p1)
    t_mul_2 = time.perf_counter()
    sum_mul = sum_mul + t_mul_2 - t_mul_1

print("Total time of 100 times tate_bilinear_pairing operation:", sum_p)
print("Average time of tate_bilinear_pairing operation:", (sum_p / 10) * 1000, 'ms')

print("Total time of 100 times scalar multiplication pairing operation:", sum_mul)
print("Average time of scalar multiplication pairing operation:", (sum_mul / 10) * 1000, 'ms')


