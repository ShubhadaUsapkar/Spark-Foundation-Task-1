#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation :Data Science and business analytics Intership
# ## Task 1:Prediction Using Supervised Machine Learning Problem
# ## Statement:Predict the percentage of an student based on the no. of study hours.
# ### Author: Nichal Akshata Naganath.

# ##  Step 1:Processing the Dataset

# In[8]:


#Loading required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


#loading the dataset
data=pd.read_csv("C:/Users/91986/Documents/student_scores - student_scores.csv")
data.head()      #Used to read first five rows of dataset


# In[10]:


data.shape #Used to find Number of rows and columns in the dataset


# In[11]:


data.info()


# In[12]:


data.describe() # This shows the descriptive statistics about the data


# In[13]:


data.isnull().sum() # To check null value and adding them


# ## Step 2:Visualization of data

# In[19]:


plt.scatter(x=data.Hours,y=data.Scores)
plt.title('Hours Vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# The above Scatter plot shows the relationship between Students Study hours and their repspctive Scores.
# From above graph we conclude that as Study hours increasing then marks also increasing

# ##  Step3:Prepare the data set for machine learning algorithm

# In[20]:


# Data Cleaning
data.mean()


# In[21]:


data2=data.fillna(data.mean())
data2.isnull().sum()


# In[22]:


#Split the dataset fpr trainig
x=data2.drop("Scores",axis="columns")
y=data2.drop("Hours",axis="columns")
print("shape of x",x.shape)
print("shape of y",y.shape)


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[24]:


print("shape of x train",x_train.shape)
print("shape of y train",y_train.shape)
print("shape of x_test",x_test.shape)
print("shape of y_test",y_test.shape)


# ## Step4:Select the model and train it

# In[28]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[29]:


lr.fit(x_train,y_train)


# In[30]:


lr.coef_


# In[31]:


lr.intercept_


# In[33]:


m=3.93
c=50.44
y=m*12+c
y


# In[35]:


lr.predict([[4]])[0][0].round(2)


# In[36]:


y_pred=lr.predict(x_test)
y_pred


# In[39]:


lr.score(x_test,y_test)


# In[40]:


plt.scatter(x_test,y_test)


# In[41]:


plt.scatter(x_test,y_test)
plt.plot(x_train,lr.predict(x_train),color='red')


# In[42]:


import joblib
joblib.dump(lr,"Students_marks_predictor.pkl")


# In[44]:


model=joblib.load("students_marks_predictor.pkl")
model.predict([[9.25]])


# If student will studied 9.25 Hours then the predicted score of student is 93.45860056
