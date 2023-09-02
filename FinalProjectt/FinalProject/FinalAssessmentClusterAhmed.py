#Done by Ahmed Amin, Adham wael, Omar Awni and Mahmoud Ibrahim (Group 1).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


########################################################################################################################
##### STEP 1 - EDA (Exploring data analysis):
df= pd.read_csv('Datasets/Sleep_health_and_lifestyle_dataset.csv')
df=pd.DataFrame(df)
#print(df.info())
#print(df.describe())




#sns.set_style("whitegrid")
#sns.pairplot(df, hue="Sleep Disorder", height =3)


#corr = df.corr()
#ax1 = sns.heatmap(corr, cbar=0, linewidths=2,vmax=1, vmin=0, square=True, cmap='Blues', annot = True)
#plt.show()


###################################################################################################################################
##### STEP 2 - Data pre-processing:


df2 = df.copy() #to be used for clustering model.

col1 = ['Occupation', 'Gender', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']
df2[col1] = df2[col1].apply(LabelEncoder().fit_transform)


df2.info()


df1 = df.copy() #To be used for RF model.
#Columns that are objects:
col = ['Occupation', 'Gender', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']
#one_hot_encoded_data = pd.get_dummies(df1, columns = ['Occupation'])
#print(one_hot_encoded_data)
df1[col] = df1[col].apply(LabelEncoder().fit_transform)

df1.drop(['BMI Category' , 'Gender' ,  'Person ID', 'Blood Pressure', 'Heart Rate', 'Stress Level'], axis = 1, inplace = True)


df1.info()

### VISUALIZATION AFTER ENCODING OBJECTS:
#sns.set_style("whitegrid")
#sns.pairplot(df1, hue="Sleep Disorder", height =3)

corr = df1.corr()
ax1 = sns.heatmap(corr, cbar=0, linewidths= 2,vmax=1, vmin=0, square=True, cmap='Blues', annot = True)

#plt.scatter(df1['Physical Activity Level'], df1['Sleep Disorder'])
#plt.xlabel('PAL')
#plt.ylabel('sleep disorder')
#plt.title('Scatter plot on PAL/sleep disorder')
plt.show()


### OBSERVATIONS:
#1 = None, 2 = sleep apnea, 0 = insomnia
#Columns to drop: 'BMI Category' , 'Gender' ,  'Person ID', 'Blood Pressure', 'Heart Rate', 'Stress Level'



########################################################################################################################################################
##### STEP 3 - Feature engineering (creating new features):

#No new features to be created for the data.


#####################################################################################################################################################
##### STEP 4 - Initiate machine learning model:
Featuresselected = df1.iloc[: , : -1] #all columns except the last one
#Featuresselected = df1.iloc[:, :] #all columns
X = Featuresselected
Y = df1.iloc[:, -1] #Only Sleep disorder (last column)

clusterX = df2 #Clustering model uses the whole dataset after changing categorical data to numerical data, as it is not predicting a particular variable.



#Coupling Random forest regressor and clustering model to PREDICT.
#If only to see patterns and associations, one should use the clustering model alone.


# What is the kmeans algorithm?
#The k-means algorithm searches for a pre-determined number of clusters within an unlabeled multidimensional dataset.
#It accomplishes this using a simple conception of what the optimal clustering looks like: The "cluster center" is the arithmetic mean of all the points belonging to the cluster.
Clustermodel = KMeans(n_clusters= 2, random_state=0, n_init="auto")
#each data point is randomly assigned to one of the K clusters.
#Then, we compute the centroid (functionally the center) of each cluster, and reassign each data point to the cluster with the closest centroid.
#We repeat this process until the cluster assignments for each data point are no longer changing.


rfmodel = RandomForestRegressor(n_estimators= 150, max_features= 7)

x_train1, x_test1, y_train1, y_test1 = train_test_split(X, Y, train_size = 0.75, test_size = 0.25) #Train, test split for RF model.


#Normalizing the data for clustering (To avoid scaling issues):
clusterX_norm = preprocessing.normalize(clusterX)

########################################################################################################################################################
##### STEP 5 - Fit the data in the model + Training:
Clustermodel.fit(clusterX_norm)


rfmodel.fit(x_train1, y_train1)

#training score:
print ("Training score for RF model:", rfmodel.score (x_train1, y_train1))

################################################################################################################################################################
##### STEP 6 - Predict:
#Predict the closest cluster each sample in X belongs to.
#In the vector quantization literature, cluster_centers_ is called the code book and each value returned by predict is the index of the closest code in the code book.
Clusters = Clustermodel.predict(clusterX_norm)
print("Cluster predictions: ", Clusters)
print("Cluster centers:", Clustermodel.cluster_centers_)

y_pred = rfmodel.predict(x_test1)

#####################################################################################################################################################################################
##### STEP 7 - Evaluation:
#To evaluate the cluster model use silhouette score. The lower the silhouette score the better the fit.
print("Silhouette score for cluster model:", silhouette_score(clusterX_norm, Clustermodel.labels_, metric='euclidean'))

#Evaluating Random forest regressor model:
print("Mean squared error for RF model:", mean_squared_error(y_pred, y_test1))
print("Testing score for RF Model:", r2_score(y_pred, y_test1))