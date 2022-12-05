#!/usr/bin/env python
# coding: utf-8

# In[31]:


#-------------------------------------------------------------------------
# AUTHOR: Ritika
# FILENAME: association_ruleAsg5Q5
# SPECIFICATION: Association rule mining
# FOR: CS 5990- Assignment #5
# TIME SPENT: 4:15 hrs
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library
#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')
df.head() 

#find the unique items all over the data and store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))
#remove nan (empty) values by using:
itemset.remove(np.nan)
#print(itemset)

#To make use of the apriori module given by mlxtend library, we need to convert the
#dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:
#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1
#To do that, create a dictionary (labels) for each transaction, store the 
#corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below
labels = {k: v for v,k in enumerate(itemset)}
#print(labels)
encoded_vals = []
for index, row in df.iterrows():
    labels = {}
    for item in itemset:
        #print(item)
        if item in row.values:
            labels[item] = 1
        else:
            labels[item] = 0
    encoded_vals.append(labels)
#print(encoded_vals)
 
#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)
#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
#print((rules))
#iterate the rules data frame and print the apriori algorithm results by using the following format:
#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
for index, rule in rules.iterrows():
    #data = frozenset(rule)
    #print(type(rule))
    #print(data.antecedents + data.consequents)
    print(','.join(list(rule.antecedents))+" -> "+','.join(list(rule.consequents)))
    print("Support:"+str(rule.support))
    print("Confidence:"+str(rule.confidence))
    print("-----------------------------------------------------")
    
#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#-->add your python code below

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

