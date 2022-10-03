#!/usr/bin/env python
# coding: utf-8

# In[13]:


# -------------------------------------------------------------------------
# AUTHOR: Ritika
# FILENAME: CS5990Asg2Q5DecisionTree
# SPECIFICATION: Building a decision tree with corresponding reading csv files
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 2hr and 15 minutes
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    #print("X",X)
    # X =
    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    #print("Y",Y)
    
    maritalStatus = {"Single":[1,0,0], "Divorced":[0,1,0], "Married":[0,0,1] }
    refund = {"Yes":1, "No":0}
    def transform_instance(instance):
        chaInsce = []
        chaInsce.append(refund[instance[0]])
        hot_encode = maritalStatus[instance[1]]
        chaInsce.append(hot_encode[0])
        chaInsce.append(hot_encode[1])
        chaInsce.append(hot_encode[2])
        taxableIncome = instance[2].replace('k','')
        taxable = float(taxableIncome)
        chaInsce.append(taxableIncome)
        return chaInsce
    
    for instance in data_training:
        chaInsce = transform_instance(instance)
        X.append(chaInsce)
        Y.append(refund[instance[3]])
    

    #loop your training and test tasks 10 times here
    testData = pd.read_csv('cheat_test.csv', sep=',', header=0)
    data_test = np.array(testData.values)[:,1:]
    accuracies = []
    for i in range (10):

        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)

        #plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        plt.show()

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for data in data_test:
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            transformedFeatures = transform_instance(data)
           
            class_predicted = clf.predict([transformedFeatures])[0]

           #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            test_class = transformedFeatures[3]
            if class_predicted == 1 and test_class == 1:
                tp+=1
            elif class_predicted == 1 and test_class == 0:
                fp+=1
            elif class_predicted == 0 and test_class == 1:
                fn+=1
            else:
                tn+=1
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        #print("ACCURACY",accuracy)
        accuracies.append(accuracy)
    print("accuracies of the model", accuracies)
       #find the average accuracy of this model during the 10 runs (training and test set)
    finalAccuracy = np.average(accuracies)

    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    print("Final accuracy when training on "+str(ds) + ': '+ str(avg_accuracy))





# In[ ]:





# In[ ]:




