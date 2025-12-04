import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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


#visualize outliers
plt.figure(figsize=(12, 12))
for i, col in enumerate(data.columns):
    #col is the column name
    #i is the index of the column
    plt.subplot(3, 3, i+1)#as matplotlib starts indexing from 1
    sns.boxplot(x = col,data = data)  


#visualize outliers but in histogram
plt.figure(figsize=(12, 12))
for i, col in enumerate(data.columns):
    plt.subplot(3, 3, i+1)
    sns.histplot(x = col,data = data, kde=True)#kde to show the curve on histogram


#plot the relationship between all numerical features and the outcome column
sns.pairplot(hue='Outcome', data = data)

#plot correlation heatmap
plt.figure(figsize=(12, 12))
sns.heatmap(data.corr(), center=0, cmap='coolwarm', annot=True, fmt='.2f')
plt.show()

#object of standard scaler
#standard scaler is used as the data has different scales
sc_x = StandardScaler()
X = pd.DataFrame(sc_x.fit_transform(data.drop('Outcome', axis=1)),#drop the outcome column because output can't be scaled
columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])#then name the columns again to make the table readable
print(X.head())
print("--------------------------------")

y = data['Outcome']

#split the data into training and testing sets in the ratio of 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



