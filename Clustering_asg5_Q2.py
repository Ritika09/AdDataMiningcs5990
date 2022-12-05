#!/usr/bin/env python
# coding: utf-8

# In[12]:


#-------------------------------------------------------------------------
# AUTHOR: Ritika
# FILENAME: Clustering_asg5_Q2
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #5
# TIME SPENT: 2:15
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library
X_training = df.to_numpy()

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
silhouette_coefficient =0
k_values = [k for k in range(2, 21)]
#for each k, calculate the silhouette_coefficient by using: 
#silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
sil_coeff_arr = []
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)
    silhouette_coefficient_value = silhouette_score(X_training, kmeans.labels_)
    sil_coeff_arr.append(silhouette_coefficient_value)
#plot the value of the silhouette_coefficient for each k value of kmeans so that we
#can see the best k
    if(silhouette_coefficient_value>silhouette_coefficient):
        silhouette_coefficient= silhouette_coefficient_value
     #--> add your Python code
plt.figure(figsize=(18, 9))
    ## sns.set_style("darkgrid")
plt.title(f'Silhouette score for different values of k)',color = 'green', fontsize=15, fontweight='bold')
plt.xlabel('K-values', fontsize=15, fontweight='bold',color = 'green')
plt.ylabel('Silhouette', fontsize=15, fontweight='bold', color = 'green')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(k_values,sil_coeff_arr,marker='o',color = 'red')
plt.show()
#reading the validation data (clusters) by using Pandas library
#Calculate and print the Homogeneity of this kmeans clustering
df = pd.read_csv('testing_data.csv', sep=",", header = None)
labels = np.array(df.values).reshape(1, len(df.index))[0]
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, 
kmeans.labels_).__str__())


# In[ ]:




