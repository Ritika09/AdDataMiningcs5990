#!/usr/bin/env python
# coding: utf-8

# In[6]:


# AUTHOR: Ritika
# FILENAME: CS5990Asg1Q10
# SPECIFICATION: Calculating cosine similarity of text
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 1:30 hrs

# Importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Defining the documents
doc1 = "Soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "I support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"

docs = [doc1, doc2, doc3, doc4]
#print(docs)

# Defining and fitting the count vectorizer on the document.
vec = CountVectorizer()
X = vec.fit_transform(docs)

# Converting the vector on the DataFrame using pandas
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
df.head()

# Using text mining
# Initializing function for making term-document matrix. 
import textmining
tdm = textmining.TermDocumentMatrix()
#print(tdm)

tdm.add_doc(doc1)
tdm.add_doc(doc2)
tdm.add_doc(doc3)
tdm.add_doc(doc4)
tdm=tdm.to_df(cutoff=0)
tdm

#words = [soccer, my, favorite, sport, I, like, one, support, olympic, game]
final_matrix = tdm.iloc[:, [0,2,3,4,5,6,9,10,13,14]]
final_matrix.index = ['doc1', 'doc2', 'doc3', 'doc4']
#print(final_matrix)
vecMatx = final_matrix
print(vecMatx)

# Compute cosine similarity
#print(cosine_similarity(final_matrix, final_matrix))

#print(cosine_similarity(vecMatx, vecMatx))
mtx = cosine_similarity(vecMatx, vecMatx)
print(mtx)

print('Seeing matrix above we see that -')
print('doc1 has heighest cosine_similarity as 0.70710678')
print('doc2 has heighest cosine_similarity as 0.72168784')
print('doc3 has heighest cosine_similarity as  0.63245553')
print('doc4 has heighest cosine_similarity as 0.72168784')

doc1Mat = [1,1,1,1,0,0,0,0,0,0]
doc2Mat = [1,1,1,0,1,1,1,0,0,0]
doc3Mat = [1,0,0,0,1,0,0,1,1,1]
doc4Mat = [1,1,1,1,1,1,0,0,1,1]
arr = []
#cosSimi1A2 = print(cosine_similarity([doc1Mat], [doc2Mat]))
cosSimi1A2 = (cosine_similarity([doc1Mat], [doc2Mat]))
#cosSimi1A3 = print(cosine_similarity([doc1Mat], [doc3Mat]))
cosSimi1A3 = cosine_similarity([doc1Mat], [doc3Mat])
#cosSimi1A4 = print(cosine_similarity([doc1Mat], [doc4Mat]))
cosSimi1A4 = cosine_similarity([doc1Mat], [doc4Mat])
#cosSimi2A1 = print(cosine_similarity([doc2Mat], [doc1Mat]))
cosSimi2A1 = cosine_similarity([doc2Mat], [doc1Mat])
#cosSimi2A3 = print(cosine_similarity([doc2Mat], [doc3Mat]))
cosSimi2A3 = cosine_similarity([doc2Mat], [doc3Mat])
#cosSimi2A4 = print(cosine_similarity([doc2Mat], [doc4Mat]))
cosSimi2A4 = cosine_similarity([doc2Mat], [doc4Mat])
#cosSimi3A4 = print(cosine_similarity([doc3Mat], [doc4Mat]))
cosSimi3A1 = cosine_similarity([doc3Mat], [doc1Mat])
#cosSimi3A2 = print(cosine_similarity([doc3Mat], [doc2Mat]))
cosSimi3A2 = cosine_similarity([doc3Mat], [doc2Mat])
#cosSimi3A4 = print(cosine_similarity([doc3Mat], [doc4Mat]))
cosSimi3A4 = cosine_similarity([doc3Mat], [doc4Mat])
#cosSimi4A1 = print(cosine_similarity([doc4Mat], [doc1Mat]))
cosSimi4A1 = cosine_similarity([doc4Mat], [doc1Mat])
#cosSimi4A3 = print(cosine_similarity([doc4Mat], [doc3Mat]))
cosSimi4A3 = cosine_similarity([doc4Mat], [doc3Mat])
#cosSimi4A2 = print(cosine_similarity([doc4Mat], [doc2Mat]))
cosSimi4A2 = cosine_similarity([doc4Mat], [doc2Mat])
arr.append(cosSimi1A2)
arr.append(cosSimi1A3)
arr.append(cosSimi1A4)
arr.append(cosSimi2A1)
arr.append(cosSimi2A3)
arr.append(cosSimi2A4)
arr.append(cosSimi3A4)
arr.append(cosSimi3A2)
arr.append(cosSimi3A1)
arr.append(cosSimi4A1)
arr.append(cosSimi4A2)
arr.append(cosSimi4A3)
result = np.amax(arr)

print('The most similar documents are: doc2 and doc4 with cosine similarity = ' + str(result))

