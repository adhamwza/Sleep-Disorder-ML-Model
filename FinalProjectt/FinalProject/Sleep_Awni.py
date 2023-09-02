import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
########################################################################################################################
##### STEP 1 - EDA (Exploring data analysis):
matplotlib.use('TkAgg')

def replace_none_with_no(df, column_name):
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        print(f"Column '{column_name}' does not exist in the DataFrame.")
        return
    #Replace None values with 'NO' in the specified column
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

df1.drop(['BMI Category' , 'Gender' , 'Person ID', 'Stress Level', 'Blood Pressure', 'Heart Rate'], axis = 1, inplace = True)


### VISUALIZATION AFTER ENCODING OBJECTS:
#sns.set_style("whitegrid")
#sns.pairplot(df1, hue="Sleep Disorder", height =3)

#corr = df1.corr()
#ax1 = sns.heatmap(corr, cbar=0, linewidths= 2,vmax=1, vmin=0, square=True, cmap='Blues', annot = True)

#plt.scatter(df1['Physical Activity Level'], df1['Sleep Disorder'])
#plt.xlabel('PAL')
#plt.ylabel('sleep disorder')
#plt.title('Scatter plot on PAL/sleep disorder')
#plt.show()


### OBSERVATIONS:
#1 = None, 2 = sleep apnea, 0 = insomnia
#A correlation between physical activity level and sleep disorder.
#A correlation between daily steps and sleep disorder. (NO other outstanding correlations for sleep disorder.)
#Columns to drop: BMI category, gender, Occupation, person ID, stress level, Blood pressure, Heart rate, sleep duration, sleep quality


#############################################################################################################################################
##### STEP 3 - Feature engineering:



#####################################################################################################################################################
##### STEP 4 - Initiate machine learning model:
x = df1.iloc[: , : -1]



y = df1.iloc[:, -1]


#regr = LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25 , random_state = 0)




########################################################################################################################################################
##### STEP 5 - Fit the data in the model + Training:
#regr.fit(x_train, y_train)

#Fitting Logistic Regression to the training set

classifier= LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)
# st_x= StandardScaler()
# x_train= st_x.fit_transform(x_train)
# x_test= st_x.transform(x_test)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=0, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)
################################################################################################################################################################
##### STEP 6 - Predict:
y_pred = classifier.predict(x_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

#####################################################################################################################################################################################
# #### STEP 7 - Evaluation:
# print(mean_squared_error(y_pred, y_test))
# print(regr.score(y_pred, y_test))
# Creating the Confusion matrix


#Visualizing the training set result

# x_set, y_set = x_train, y_train
# x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),
# np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
# alpha = 0.75, cmap = ListedColormap(('purple','green' )))
#
# plt.xlim(x1.min(), x1.max())
# plt.ylim(x2.min(), x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
#         c = ListedColormap(('purple', 'green'))(i), label = j)
# plt.title('Logistic Regression (Training set)')
# plt.xlabel('Physical Activity Level')
# plt.ylabel('Daily Steps')
# plt.legend()
# plt.show()





#
# # name  of classes
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
 # create heatmap
df2 = pd.DataFrame(cnf_matrix)
sns.heatmap(df2, cbar=0, linewidths= 2,vmax=1, vmin=0, square=True, cmap='Blues', annot = True)
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()




# plt.figure(figsize= (6, 4))
# sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g')
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.show()




