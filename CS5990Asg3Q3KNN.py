#!/usr/bin/env python
# coding: utf-8

# In[1]:


#-------------------------------------------------------------------------
# AUTHOR: Ritika
# FILENAME: CS590Asg3Q3KNN.py
# SPECIFICATION: Grid search for KNN multiple values hyperparameters
# FOR: CS 5990- Assignment #3
# TIME SPENT: 4 hrs 15 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
training_data = pd.read_csv('weather_training.csv')
#print(train_data.head(), "OK")
#print(train_data.head())
y_training = np.array(training_data["Temperature (C)"])
#print(y_train_data)
#y_train_data
X_training = np.array(training_data.drop(["Temperature (C)", "Formatted Date"], axis=1).values)
#X_train_data

#reading the test data
test_data = pd.read_csv('weather_test.csv')
#print(test_data.head())
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
X_test = test_data.drop(["Temperature (C)", "Formatted Date"], axis=1).values
y_test = test_data["Temperature (C)"]

#loop over the hyperparameter values (k, p, and w) ok KNN
#--> add your Python code here
accuracy_result = 0
for k in k_values:
    for p in p_values:
        for w in w_values:
            count_accu = 0
            #fitting the knn to the data
            clf = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, y_training)
            
            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                predicted_value = clf.predict(np.array([x_testSample]))
                #print(predicted_value)
                #the prediction should be considered correct if the output value is [-15%,+15%] 
                #distant from the real output values.
                #to calculate the % difference between the prediction and the real output values
                #use: 100*(|predicted_value - real_value|)/real_value))
                diff = 100*(abs(predicted_value[0] - y_testSample)/y_testSample)
                #print(diff)
                if diff <=15 and diff >= -15:
                    count_accu +=  1
                #print(count_accu)
                #check if the calculated accuracy is higher than the previously one calculated. 
                #If so, update the highest accuracy and print it together
                #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 
                #0.92, Parameters: k=1, p=2, w= 'uniform'"
            result = count_accu/len(y_test)
            if result > accuracy_result:
                print(f"Highest KNN accuracy so far: {result}, Parameters: k={k}, p={p}, w= '{w}'")
                accuracy_result = result


# In[ ]:




