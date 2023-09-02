#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# # STEP 1 - EDA (Exploring data analysis):

# In[2]:


df= pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
df=pd.DataFrame(df)
print(df.info())
print("--------------------------------------------------------------------------------------")
print(df.describe())


# In[3]:


sns.set_style("whitegrid")
sns.pairplot(df, hue="Sleep Disorder", height =3)


# In[4]:


corr = df.corr()
ax1 = sns.heatmap(corr, cbar=True, linewidths=2,vmax=1, square=True, cmap='coolwarm', annot = True)
plt.show()


# # STEP 2 - Data pre-processing:

# In[5]:


df1 = df.copy()


# ### Columns that are objects:

# In[6]:


col = ['Occupation', 'Gender', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']


# In[7]:


df1[col] = df1[col].apply(LabelEncoder().fit_transform)


# In[8]:


df1.drop(['BMI Category' , 'Gender' , 'Person ID', "Stress Level", "Blood Pressure", "Heart Rate"], axis = 1, inplace = True)


# ### VISUALIZATION AFTER ENCODING OBJECTS:

# In[9]:


sns.set_style("whitegrid")
sns.pairplot(df1, hue="Sleep Disorder", height =3)


# In[10]:


corr2 = df1.corr()
ax1 = sns.heatmap(corr2, cbar=True, linewidths= 2,vmax=1, vmin=0, square=True, cmap='Blues', annot = True)


# ### OBSERVATIONS:

# - 1 = None, 2 = sleep apnea, 0 = insomnia
# - A correlation between physical activity level and sleep disorder.
# - A correlation between daily steps and sleep disorder. (NO other outstanding correlations for sleep disorder.)
# - Columns to drop: BMI category, gender, Occupation, person ID, stress level, Blood pressure, Heart rate, sleep duration, sleep quality

# # STEP 3 - Feature Selection:

# In[11]:


X = df1.drop('Sleep Disorder', axis=1)
y = df1['Sleep Disorder']


# # STEP 4 - Initiate machine learning model:

# In[12]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# # STEP 5 - Fit the data in the model + Training:

# In[13]:


k = 5  #You can set the value of K to any positive integer (usually an odd number)

knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)


# # STEP 6 - Predict:

# In[14]:


y_pred = knn_model.predict(X_test)


# # STEP 7 - Evaluation:

# In[15]:


# accuracy using the KNN Algorithm
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# ### Vizualizing the KNN algorithm
# 

# The numbers inside the heatmap show how many samples from each class the KNN model successfully or erroneously predicted.

# In[16]:


def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

plot_cm(y_test, y_pred)

