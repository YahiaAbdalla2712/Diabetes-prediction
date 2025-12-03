import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#read the data from the csv file
#in the outcome column, 1 means the patient has diabetes, 0 means the patient does not have diabetes
data = pd.read_csv('diabetes.csv')#this data is from kaggle

print(data.head())
print("--------------------------------")

print(data.info())#to know how many null values are there in the data
print("--------------------------------")

#Or we can use .isna().sum() to know how many null values are there in the data
print(data.isna().sum())
print("--------------------------------")

#check for duplicate values
print(data.duplicated().sum())
print("--------------------------------")

#visualize the data
plt.figure(figsize=(12, 6))
sns.countplot(x='Outcome', data=data)#to plot the count of the outcome column
plt.show()


#visualize outliers
plt.figure(figsize=(12, 12))
for i, col in enumerate(data.columns):
    #col is the column name
    #i is the index of the column
    plt.subplot(3, 3, i+1)#as matplotlib starts indexing from 1
    sns.boxplot(x = col,data = data)
plt.show()   


#plot the relationship between all numerical features and the outcome column
sns.pairplot(hue='Outcome', data = data)
plt.show()
