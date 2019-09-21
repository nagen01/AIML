# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 08:34:46 2019

@author: Nagendra
"""
#Upper confidence bound
#Importing important libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the CSV file
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Preparing the model
import math
N = 10000
d = 10
number_of_selections = [0]*d
sum_of_rewards = [0]*d
ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if (number_of_selections[i] > 0):
            average_reward = sum_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(1.5*math.log(n+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400    
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    sum_of_rewards[ad] =sum_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
plt.hist(ads_selected)
plt.title('Histogram for add sellection')
plt.xlabel('Ads')
plt.ylabel('Number of times add sellected')
plt.show()