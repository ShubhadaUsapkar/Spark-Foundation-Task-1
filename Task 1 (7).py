# ## Task 1:Prediction Using Supervised Machine Learning Problem
# ## Statement:Predict the percentage of an student based on the no. of study hours.
# ## Author: Usapkar Shubhada Mangesh
# ##  Step 1:Processing the Dataset
# Loading required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#loading the dataset
data=pd.read_csv("C:/Users/Viraj Vijay Samant/Downloads/task 1 data.csv")
data.head()   #Used to read first five rows of dataset
data.shape #Used to find Number of rows and columns in the dataset
data.info()
data.describe() # This shows the descriptive statistics about the data
data.isnull().sum() # To check null value and adding them
# ### Step 2:Visualization of data
plt.scatter(x=data.Hours,y=data.Scores)
plt.title('Hours Vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()
# ### The above Scatter plot shows the relationship between Students Study hours and their repspctive Scores.
# ### From above graph we conclude that as Study hours increasing then marks also increasing
# ##  Step3:Prepare the data set for machine learning algorithm
# Data Cleaning
data.mean()
data2=data.fillna(data.mean())
data2.isnull().sum()
#Split the dataset fpr trainig
x=data2.drop("Scores",axis="columns")
y=data2.drop("Hours",axis="columns")
print("shape of x",x.shape)
print("shape of y",y.shape)
 from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print("shape of x train",x_train.shape)
print("shape of y train",y_train.shape)
print("shape of x_test",x_test.shape)
print("shape of y_test",y_test.shape)
# ### Step4:Select the model and train it
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
lr.coef_
lr.intercept_
m=3.93
c=50.44
y=m*12+c
y
lr.predict([[4]])[0][0].round(2)
y_pred=lr.predict(x_test)
y_pred
lr.score(x_test,y_test)
plt.scatter(x_test,y_test)
plt.scatter(x_test,y_test)
plt.plot(x_train,lr.predict(x_train),color='red')
import joblib
joblib.dump(lr,"Students_marks_predictor.pkl")
model=joblib.load("students_marks_predictor.pkl")
model.predict([[9.25]])
#  If student will studied 9.25 Hours then the predicted score of student is 93.45860056
