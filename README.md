# SleepGuard: Predicting Sleep Disorders with ML

## Description:
This project's purpose is to predict whether a person will have a sleep disorder or not while also predicting the type of sleep disorder, whetehr it is sleep apnea, insomnia, and more using multiple ML models.


## Step 1 - Exploratory Data Analysis (EDA):
The programme starts by importing the required libraries and extracting the dataset from a CSV file with the file extension "Sleep_Health_and_Lifestyle_Dataset.csv." subsequently basic dataset details like the data types and non-null counts for each column are printed. The 'describe()' method is also used to display an overview of statistical measurements for numeric columns. The dataset has 374 rows and 13 columns with objects (strings), integers, and floats among other data types.

## Step 1 (continued) - Data Visualization:
Following the first data investigation, the algorithm use seaborn to provide data visualisations. The dataset's correlations between various numerical variables, classified by the "Sleep Disorder" feature, are first visualised using a pair plot. Each graphic demonstrates the relationships between several variables and the distribution of data points according to various sleep disorders.

Furthermore, The correlation matrix between the dataset's numerical characteristics is then shown as a heatmap. Colours are used in the correlation heatmap to show the direction and degree of the link between features. The code also inserts numerical numbers to display the correlation coefficients within every cell. 

## Step 2 - Data Pre-processing:
To maintain the original data, the code generates a copy of the original dataframe named "df1" in this phase. The categorical variables are subsequently transformed into numeric values using labelled encoding. Occupation, Gender, BMI Category, Blood Pressure, and Sleep Disorder are the columns that are being encoded.

The algorithm eliminates a number of columns that are assessed unnecessary for the categorization operation after encoding the category columns. BMI Category, Gender, Person ID, Stress Level, Blood Pressure, and Heart Rate are the columns that are being removed.

## Step 2 (continued) - Data Visualization after Encoding:
Afterwards,  pre-processing the data, the algorithm creates fresh visualisations to investigate connections among the characteristics and the intended variable, "Sleep Disorder." To display the encoded characteristics and their correlations with the target variable, it generates another pair plot and a correlation heatmap.

## Step 3 - Feature Selection:
In this stage, the code divides the characteristics from the target variable "Sleep Disorder" and assigns them to X and y, accordingly. This is a standard machine learning strategy where the target variable is used for prediction while the characteristics are utilised to train the model.

## Model Explanation:

## K-Nearest Neighbors (KNN)
A straightforward and user-friendly supervised machine learning technique called K-Nearest Neighbours (KNN) is employed for classification and regression applications. The fundamental goal of KNN is to categorise a data point based on the feature space's k closest neighbours' predominant class. In other words, it makes the assumption that comparable data points are probably members of the same class.

KNN's basic operating premise is as follows:
Data Collection
Distance Calculation
Selecting K
Voting and Classification
Evaluation

It's important to note that KNN is a non-parametric and lazy learning algorithm, which means it doesn't make any assumptions about the distribution of the underlying data and doesn't build an explicit model during training. Instead, the complete dataset is stored, and calculations are made when a prediction is made. While KNN is straightforward and simple to use, its computational cost can drastically rise with big datasets, making it less appropriate for real-time applications or high-dimensional data. KNN is still a useful and popular approach, especially for smaller datasets and as a comparison tool for more advanced machine learning models.

Training data to the KNN model using the `fit()` function, which trains the model based on the provided data.


## Random Forest model
The decision trees, which resemble flowcharts, on which the Random Forest model is built. Each leaf node represents a class label (for classification) or a numerical value (for regression), whereas each internal node represents a feature and each branch a choice based on that feature. Decision trees use feature values to build pure subsets of data that only contain instances of the same class.

Using bootstrapping, Random Forest generates several decision trees from replacement samples taken from random subsets of the training data. 

Random feature selection is another important part of Random Forest. By dividing each node in the decision trees using only a random subset of features, it adds more randomization to the system. As a result, the variety of the trees is increased and overfitting is prevented.

Explanation: broke down the data into training and testing sets x train, x test, and the test size was 20% with and random state of 42. Function is: random state( random sampling)

## Logistic regression
I utilise the statistical approach logistic regression to solve binary classification issues. It comes in helpful when I need to forecast results with just two classes or alternatives. I'm interested in predicting categorical outcomes, which are often expressed as 0 or 1, rather than numerical values as in linear regression.

The function I used to train a logistic regression model in Python's scikit-learn module is named LogisticRegression. This is how I apply it: So now that I've created a logistic regression model instance with LogisticRegression(), my model is prepared to learn from the training set. I give it my feature matrix (X_train), which represents the input variables, and the associated target vector (Y_train), which represents the class labels or results.Throughout this training process, the model learns how features and labels relate to one another. After I've trained my model, I can use it to predict outcomes based on fresh, unforeseen data:The predicted class labels for those samples are returned by the model as an array of 0s and 1s in the variable y_pred when I use the predict function with the new data (feature matrix) X_test.

When there is a roughly linear relationship between the characteristics and the outcome's log-odds, logistic regression is effective. It's a useful algorithm for many uses, including sentiment analysis, illness diagnosis, and spam identification.

## K-Means Cluster + Random Forest Regressive 
When initialising the model, the predetermined number of clusters is defined by code, and the k-means clustering algorithm looks for those clusters. Particularly one of its characteristics, n_clusters, which refers to the number of clusters in a multidimensional dataset without labels. It does this by employing a straightforward definition of what the ideal clustering looks like: The "cluster centre" is the mathematical mean of all the cluster's points. Each data point is sorted into one of the K clusters at random. After that, we calculate each cluster's centroid (which is essentially its centre) and reassign each data point to the cluster with the closest centroid. This procedure is repeated until there is no longer any change in the cluster assignments for any given data point. 

In essence, clustering looks for recurring patterns and relationships between characteristics and their the dataset's pertinent data. The nearest cluster to which each sample in X belongs is predicted by using the predict option in the k-means clustering model. The same dataset that you used to train the model is what you enter. Each number provided by predict is the index of the nearest code in the code book, which is referred to in the vector quantization literature as cluster_centers_. The silhouette score must be used to assess the cluster model. The better the fit, the lower the silhouette score. 

A random forest regressor model was utilised in conjunction with the clustering model to accomplish our machine learning project's primary goal of predicting whether a person has a sleeping issue or not.


## Note:
Don't forget to include the CSV file in the code folder
