#!/usr/bin/env python
# coding: utf-8

# In[43]:


# -------------------------------------------------------------------------
# AUTHOR: Ritika
# FILENAME: CS5990Asg2Q6ROCcurve
# SPECIFICATION: Plotting ROC curve based on decision tree
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# read the dataset cheat_data.csv and prepare the data_training numpy array
df = pd.read_csv('cheat_data.csv', sep=',', header=0)
maritalStatus = {'Single':[1,0,0], 'Divorced':[0,1,0], 'Married':[0,0,1] }
refund = {'Yes':1, 'No':0}   
df['Marital Status'] = df['Marital Status'].map(maritalStatus)
df['Refund'] = df['Refund'].map(refund)
df['Taxable Income'] = df['Taxable Income'].str.replace('k', '')
df['Taxable Income'] = df['Taxable Income'].astype(float)
df['Cheat'] = df['Cheat'].map(dict(Yes=1, No=0))
#print(df)
    
data_training = np.array(df.values)[:,:-1] #creating a training matrix without the id (NumPy library)
#print(data_training)

# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
# --> add your Python code here
# Separating data into X and Y
def sepXandY(data):
    data = np.insert(data, 1, data[1])
    data = np.delete(data,4)
    return data
data_training = np.apply_along_axis(sepXandY, axis=1, arr=data_training)
#print(data_training)
X = data_training
Y = df.Cheat
#print(Y)

#Spliting the data set into training and testing
trainX, testX, trainy, testy = train_test_split(X, Y, test_size = 0.3)


# generate a no skill prediction (random classifier - scores should be all zero)
# --> add your Python code here
ns_probs = [0 for _ in range(len(testy))]

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)
print(dt_probs)

# keep probabilities for the positive outcome only
# --> add your Python code here
dt_probs = dt_probs[:,1]

# calculate scores by using both classifeirs (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()


# In[ ]:




