import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


##### STEP 1 - EDA (Exploring data analysis):
def replace_none_with_no(df, column_name):
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        print(f"Column '{column_name}' does not exist in the DataFrame.")
        return
    
    # Replace None values with 'NONE' in the specified column
    #df[column_name]: This selects the specified column from the DataFrame.
    #apply(): This applies a function to each element in the column.
    #lambda x: 'NONE' if pd.isna(x) else x: This is a lambda function that takes an element x from the column as input. 
    # It checks if the element is None using pd.isna(x). If it is None, it returns 'NONE', otherwise, it returns the original value of x.
    df[column_name] = df[column_name].apply(lambda x: 'NONE' if pd.isna(x) else x)
    return df


df= pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
df = replace_none_with_no(df, 'Sleep Disorder')
df=pd.DataFrame(df)
print(df.info())
print(df.describe())




#sns.set_style("whitegrid")
#sns.pairplot(df, hue="Sleep Disorder", height =3)


#corr = df.corr()
#ax1 = sns.heatmap(corr, cbar=0, linewidths=2,vmax=1, vmin=0, square=True, cmap='Blues', annot = True)
#plt.show()


###################################################################################################################################
##### STEP 2 - Data pre-processing:
df1 = df.copy()



#Columns that are objects:
col = ['Occupation', 'Gender', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']




#one_hot_encoded_data = pd.get_dummies(df1, columns = ['Occupation'])
#print(one_hot_encoded_data)


df1[col] = df1[col].apply(LabelEncoder().fit_transform)

df1.drop(['BMI Category' , 'Gender' , 'Occupation', 'Person ID', 'Stress Level', 'Blood Pressure', 'Heart Rate', 'Sleep Duration', 'Quality of Sleep', 'Age'], axis = 1, inplace = True)
df1.drop(df1[df1['Daily Steps']> 9500].index , axis = 0, inplace = True )

### VISUALIZATION AFTER ENCODING OBJECTS:
sns.set_style("whitegrid")
sns.pairplot(df1, hue="Sleep Disorder", height =3)

df1.info()

#corr = df1.corr()
#ax1 = sns.heatmap(corr, cbar=0, linewidths= 2,vmax=1, vmin=0, square=True, cmap='Blues', annot = True)

#plt.scatter(df1['Physical Activity Level'], df1['Sleep Disorder'])
#plt.xlabel('PAL')
#plt.ylabel('sleep disorder')
#plt.title('Scatter plot on PAL/sleep disorder')
plt.show()


### OBSERVATIONS:
#1 = None, 2 = sleep apnea, 0 = insomnia
#A correlation between physical activity level and sleep disorder.
#A correlation between daily steps and sleep disorder. (NO other outstanding correlations for sleep disorder.)
#Columns to drop: BMI category, gender, Occupation, person ID, stress level, Blood pressure, Heart rate, sleep duration, sleep quality


#############################################################################################################################################
##### STEP 3 - Feature engineering:

#Finished w step 2



#####################################################################################################################################################
##### STEP 4 - Initiate machine learning model:
#x = df1.iloc[: , : -1]

# Separate the target variable (y) and the features (X), so x = anything but sleep disorder
X = df1.drop('Sleep Disorder', axis=1)
y = df1['Sleep Disorder']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#y = df1.iloc[:, -1]


#regr = LinearRegression()

#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, test_size = 0.25)




########################################################################################################################################################
##### STEP 5 - Fit the data in the model + Training:
#regr.fit(x_train, y_train)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier on the training data
rf_classifier.fit(X_train, y_train)



################################################################################################################################################################
##### STEP 6 - Predict:
#y_pred = regr.predict(x_test)
y_pred = rf_classifier.predict(X_test)

print(y_pred)



#####################################################################################################################################################################################
##### STEP 7 - Evaluation:
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
#print(mean_squared_error(y_pred, y_test))
#print(regr.score(y_pred, y_test))

