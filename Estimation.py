#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-11-12
@LastEditor: JK211
LastEditTime: 2024-4-15
@Discription:  This code fully implements the function of AEABIR, we set the illegal request ratio behavior of the
attacker to increase linearly first, and then change randomly later, and tested the performance of the algorithm,
including the relative error, the weight distribution graphs, and so on. The experimental graphs output from
this code correspond to Figures 4 and 5 of the paper, and then since AEABIR is a probabilistic estimation algorithm,
there is a slight error in each experiment, which is a normal phenomenon.
@Environment: python 3.8
'''
import numpy as np
import pandas as pd
import random as rand
from statsmodels.tsa.holtwinters import Holt
from matplotlib import pyplot as plt
import matplotlib
import csv
matplotlib.rc("font", family='YouYuan')
from sklearn.metrics import mean_squared_error


def batch_request_generating(batch_size, invalid_size):
    Sigs2beVefi = np.zeros(batch_size, int)
    for i in range(invalid_size):
        positive_position = rand.randrange(batch_size)

        while Sigs2beVefi[positive_position] == 1:
            positive_position = rand.randrange(batch_size)

        Sigs2beVefi[positive_position] = 1

    return Sigs2beVefi


# Predictive analysis
def forecast(hisResult):

    if(len(hisResult) == 0):
        p = 0
    elif(len(hisResult) == 1):
        p = hisResult[0]
    else:
        y3 = pd.Series(hisResult)
        ets3 = Holt(y3)
        r3 = ets3.fit()
        p = r3.predict(start=len(hisResult), end=len(hisResult))[len(hisResult)]
    return p


# Sampling analysis
def sampling(data, subsetNum):
    # np.random.shuffle(data)
    subset = []
    for i in range(subsetNum):
        subset.append(data[i])

    count = 0
    for i in range(subsetNum):
        if(subset[i] == 1):
            count = count + 1
    return count/subsetNum


# ++++++++++++++++++ Adaptive weight calculation function （Version 3）++++++++++++++++++++++++++++++
# This code corresponds to equation (9) inside the paper
# p[i] means p_i, hisResult[i] means r_i, s[len(s)-1] means m_t, len(hisResult) means l
def weight_cal(hisResult, p, s):
    if(len(hisResult) == 0):
        a = 0
    else:
        a_deno1 = 0
        for i in range(len(hisResult)):
            a_deno1 = a_deno1 + (abs(p[i] - hisResult[i]) / hisResult[i])
        a_deno1_avg = a_deno1 / len(hisResult)
        if s[len(s) - 1] != 0:
            a_deno2 = abs(p[len(p) - 1] - s[len(s) - 1]) / s[len(s) - 1]
            temp = 1 / (a_deno1_avg + a_deno2)
            a = np.exp(-1 / temp)
        else:
            a = 0
    return a


# An alternate function, which has a slightly different effect than V3 above, was not ultimately used in the paper.
# def weight_cal(hisResult, p, s):
#     if(len(hisResult) == 0):
#         a = 0
#     else:
#         a_deno1 = 0
#         for i in range(len(hisResult)):
#             a_deno1 = a_deno1 + (abs(p[i] - hisResult[i]) / hisResult[i])
#         # a_deno1_avg = a_deno1 / len(hisResult)
#         a_deno2 = abs(p[len(p) - 1] - s[len(s) - 1]) / s[len(s) - 1]
#         a_deno_avg = (a_deno1 + a_deno2) / (len(hisResult) + 1)
#         temp = 1 / a_deno_avg
#         a = np.exp(-1 / temp)
#     return a


hisResult = []  # Historical results
p = []  # Predicted values set
s = []  # Sampling values set
esti_result = []  # Estimated values set
alpha = [] # predicted value weight
beta = []  # sampling value weight
p_inaccu = []
s_inaccu = []
e_inaccu = []   # Error between estimated and true values

# Attacker's behavioral setup, corresponding to subsection 6.3 in the paper, for example 130/2000=0.065
att = [130, 160, 190, 220, 250, 280, 310, 340, 370, 400, 430, 460, 490, 520, 550, 580, 610, 640, 670, 700,
       380, 270, 450, 100, 80, 30, 400, 550, 500, 730, 380, 270, 150, 300, 80, 10, 560, 380, 120, 360]


for i in range(len(att)):
    batch_data = batch_request_generating(2000, att[i])
    # 1) Predictive Analysis
    p_t = forecast(hisResult)
    p.append(p_t)
    # 2) Sampling Analysis
    np.random.shuffle(batch_data)
    s_t_1 = sampling(batch_data, 42)
    s.append(s_t_1)

    #  3) Complementary Sampling
    if s[i] != 0:
        if (abs(p[i]-s[i])/s[i]) > 0.7:  # The value greater than here is a threshold set manually
            s_t_2 = sampling(batch_data, 92)
            s[i] = s_t_2
    else:
        pass

    #  4) Calculation of Predicted and Sampling Weights
    alpha.append(weight_cal(hisResult, p, s))
    beta.append(1 - alpha[i])

    #  5) Weighted Sum Calculation for Estimated Value
    e = alpha[i] * p[i] + beta[i] * s[i]
    esti_result.append(e)

    p_ina = (p[i] - (att[i]/2000)) / (att[i]/2000)
    p_inaccu.append(p_ina)

    s_ina = (s[i] - (att[i] / 2000)) / (att[i] / 2000)
    s_inaccu.append(s_ina)

    e_ina = (esti_result[i] - (att[i]/2000)) / (att[i]/2000)
    e_inaccu.append(e_ina)

    hisResult.append(att[i] / 2000)

# Two stages, increase + random fluctuation
p_mse_1 = mean_squared_error(p[0:19], hisResult[0:19])
s_mse_1 = mean_squared_error(s[0:19], hisResult[0:19])
e_mse_1 = mean_squared_error(esti_result[0:19], hisResult[0:19])
# Mean Square Error (MSE)
print("The MSE of the predicted values is:", p_mse_1)
print("The MSE of the sampling values is:", s_mse_1)
print("The MSE of the estimated values is:", e_mse_1)

data_1 = [p_mse_1, s_mse_1, e_mse_1]

p_mse_2 = mean_squared_error(p[20:39], hisResult[20:39])
s_mse_2 = mean_squared_error(s[20:39], hisResult[20:39])
e_mse_2 = mean_squared_error(esti_result[20:39], hisResult[20:39])
print("The MSE of the predicted values is:", p_mse_2)
print("The MSE of the sampling values is:", s_mse_2)
print("The MSE of the estimated values is:", e_mse_2)

data_2 = [p_mse_2, s_mse_2, e_mse_2]

p_mse = mean_squared_error(p, hisResult)
s_mse = mean_squared_error(s, hisResult)
e_mse = mean_squared_error(esti_result, hisResult)

labels = ['Increase', 'Fluctuation', 'Average']
p_values = [p_mse_1, p_mse_2, p_mse]
s_values = [s_mse_1, s_mse_2, s_mse]
e_values = [e_mse_1, e_mse_2, e_mse]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, p_values, width, label='Predicted values')
rects2 = ax.bar(x, s_values, width, label='Sampling values')
rects3 = ax.bar(x + width, e_values, width, label='Estimated values')

# Draw mean-square error graph, corresponding to Fig. 4(b) on page 11 of manuscript
ax.set_ylabel('Mean-square error values')
ax.set_title('Comparison of prediction, sampling and estimated MSE values')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

data = []
data.append(p_values)
data.append(s_values)
data.append(e_values)
header = ['Predicted values', 'Sampling values', 'Estimated values']
with open('mse_data.csv', 'w', newline='') as file:
    wri = csv.writer(file)
    wri.writerow(header)
    wri.writerows(data)


# Draw a set of graphs showing the proportion of illegal requests,
# corresponding to Fig. 3(a)(b)(c) on page 10 of manuscript
plt.figure(figsize=(10, 8), dpi=300)
ax1 = plt.subplot(2, 2, 1)
plt.plot(hisResult, color='red', marker='o', linestyle='dashed', label='True values')
plt.plot(p, color='yellowgreen', marker='d', linestyle='dashed', label='Predicted values')
plt.xlabel('Rounds of batch identification')
plt.ylabel('Percentage of illegal requests (%)')
plt.title("Accuracy comparision of True and Predicted values")
plt.legend()

ax2 = plt.subplot(2, 2, 2)
plt.plot(hisResult, color='red', marker='o', linestyle='dashed', label='True values')
plt.plot(s, color='dodgerblue', marker='P', linestyle='dashed', label='Sampling values')
plt.xlabel('Rounds of batch identification')
plt.ylabel('Percentage of illegal requests (%)')
plt.title("Accuracy comparision of True and Sampling values")
plt.legend()

ax3 = plt.subplot(2, 2, 3)
plt.plot(hisResult, color='red', marker='o', linestyle='dashed', label='True values')
plt.plot(esti_result, color='gold', marker='s', linestyle='dashed', label='Estimated values')
plt.xlabel('Rounds of batch identification')
plt.ylabel('Percentage of illegal requests (%)')
plt.title("Accuracy comparision of True and Estimated values")
plt.legend()

ax4 = plt.subplot(2, 2, 4)
plt.plot(hisResult, color='red', marker='o', linestyle='dashed', label='True values')
plt.plot(p, color='yellowgreen', marker='d', linestyle='dashed', label='Predicted values')
plt.plot(s, color='dodgerblue', marker='P', linestyle='dashed', label='Sampling values')
plt.plot(esti_result, color='gold', marker='s', linestyle='dashed', label='Estimated values')
plt.xlabel('Rounds of batch identification')
plt.ylabel('Percentage of illegal requests (%)')
plt.title("Accuracy comparision of All values")
plt.legend()

plt.show()


data = []
data.append(hisResult)
data.append(p)
data.append(s)
data.append(esti_result)
data_array = np.array(data)
header = ['True values', 'Predicted values', 'Sampling values', 'Estimated values']
with open('estimate_data.csv', 'w', newline='') as file:
    wri = csv.writer(file)
    wri.writerow(header)
    wri.writerows(data_array.T)


# Draw weight change graph, corresponding to Fig. 4(c) on page 11 of manuscript
plt.figure(dpi=300)
plt.plot(alpha, color='dodgerblue', marker='o', linestyle='dashed', label='Alpha values')
plt.plot(beta, color='darkviolet', marker='d', linestyle='dashed', label='Beta values')
plt.xlabel('Rounds of batch identification')
plt.ylabel('Weight values')
plt.title("Weight distribution")
plt.legend()
plt.show()

data = []
data.append(alpha)
data.append(beta)
data_array = np.array(data)
header = ['alpha values', 'beta values']
with open('weights_data.csv', 'w', newline='') as file:
    wri = csv.writer(file)
    wri.writerow(header)
    wri.writerows(data_array.T)

# Draw a set of error graphs, corresponding to Fig. 4(a) on page 11 of manuscript
plt.figure(figsize=(10, 10), dpi=300)

p_inaccu_percent = [i*100 for i in p_inaccu]
ax1 = plt.subplot(3, 1, 1)
plt.plot(p_inaccu_percent, color='dodgerblue', marker='o', linestyle='dashed', label='Predicted Error rates')
# ax1 = brokenaxes(ylims=((0, 100), (1500, 2000)), despine=False, hspace=0.05, d=0.01)
plt.xlabel('Rounds of batch identification')
plt.ylabel('Relative error (%)')
plt.title("Error of Predicted values")
plt.legend()

s_inaccu_percent = [i*100 for i in s_inaccu]
ax2 = plt.subplot(3, 1, 2)
plt.plot(s_inaccu_percent, color='gold', marker='o', linestyle='dashed', label='Sampling Error rates')
plt.xlabel('Rounds of batch identification')
plt.ylabel('Relative error (%)')
plt.title("Error of Sampling values")
plt.legend()

e_inaccu_percent = [i*100 for i in e_inaccu]
ax3 = plt.subplot(3, 1, 3)
plt.plot(e_inaccu_percent, color='darkviolet', marker='o', linestyle='dashed', label='Estimated Error rates')
plt.xlabel('Rounds of batch identification')
plt.ylabel('Relative error (%)')
plt.title("Error of Estimated values")
plt.legend()

plt.rcParams['axes.unicode_minus'] = False  # Used to display negative signs normally
plt.show()

count_p = 0
count_s = 0
count_e = 0
for item in p_inaccu_percent:
    if item >= -20 and item <= 20:
        count_p = count_p + 1

for item in s_inaccu_percent:
    if item >= -20 and item <= 20:
        count_s = count_s + 1

for item in e_inaccu_percent:
    if item >= -20 and item <= 20:
        count_e = count_e + 1

print("The number of points that meet the condition for the positive and negative error of +-20% of the predicted values is:", count_p)
print("The number of points that meet the condition for the positive and negative error of +-20% of the sampling values is:", count_s)
print("The number of points that meet the condition for the positive and negative error of +-20% of the estimated values is:", count_e)

count_p = 0
count_s = 0
count_e = 0
for item in p_inaccu_percent:
    if item >= -15 and item <= 15:
        count_p = count_p + 1

for item in s_inaccu_percent:
    if item >= -15 and item <= 15:
        count_s = count_s + 1

for item in e_inaccu_percent:
    if item >= -15 and item <= 15:
        count_e = count_e + 1

print("The number of points that meet the condition for the positive and negative error of +-15% of the predicted values is:", count_p)
print("The number of points that meet the condition for the positive and negative error of +-15% of the sampling values is:", count_s)
print("The number of points that meet the condition for the positive and negative error of +-15% of the estimated values is:", count_e)

data = []
data.append(p_inaccu_percent)
data.append(s_inaccu_percent)
data.append(e_inaccu_percent)
data_array = np.array(data)
header = ['Predicted Error rates', 'Sampling Error rates', 'Estimated Error rates']
with open('error_data.csv', 'w', newline='') as file:
    wri = csv.writer(file)
    wri.writerow(header)
    wri.writerows(data_array.T)