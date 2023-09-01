import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#Lets solve this using Logistic Regression and Support Vector Machine Algorithms

#Importing Dataset
data = pd.read_csv('iris.csv')

#Check out the length of dataset which provides us the number of rows
print(len(data))
#Check out the head and names of columns of the dataset
print(data.head())
print(data.columns)
#Check out the info of dataset
print(data.info())
#Check out the statistical values of the dataset
print(data.describe())

#DATA VISUALIZATION USING SEABORN

sns.pairplot(data,hue="species")
plt.show()

#NULL VALUES
print(data.isna())
print(data.isna().sum())

print(data.dtypes)


#Extracting Dependent and Independent Variables
x=data[['sepal_length','sepal_width','petal_length','petal_width']]
y=data[['species']]

#SPLITTING DATASET
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#LOGISTIC REGRESSION
reg=LogisticRegression()
reg.fit(x_train,y_train)
predict_1=reg.predict(x_test)

#CLASSIFICATION REPORT
print(classification_report(y_test,predict_1))

#SUPPORT VECTOR MACHINE
s=SVC()
s.fit(x_train,y_train)
predict=s.predict(x_test)

#CLASSIFICATION REPORT
print(classification_report(y_test,predict))



