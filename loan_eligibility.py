import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

dataset = pd.read_csv("ltrain.csv")

dataset.head()

dataset.shape

dataset.info

dataset.describe()

pd.crosstab(dataset['Credit_History'], dataset['Loan_Status'], margins=True)

dataset.boxplot(column="ApplicantIncome")

dataset['ApplicantIncome'].hist(bins=20)

dataset['CoapplicantIncome'].hist(bins=20)

dataset.boxplot(column='ApplicantIncome', by='Education')

dataset.boxplot(column='LoanAmount')

dataset['LoanAmount'].hist(bins=20)

dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)

dataset.isnull().sum()

dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)

dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)

dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)

dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)

dataset.LoanAmount=dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log=dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())

dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)

dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)

dataset.isnull().sum()

dataset['TotalIncome']=dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome_log']=np.log(dataset['TotalIncome'])

dataset['TotalIncome_log'].hist(bins=20)

dataset.head()

x=dataset.iloc[:, np.r_[1:5, 9:11, 13:15]].values
y=dataset.iloc[:, 12].values

x

y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0)

print(x_train)

from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()

for i in range(0,5):
  x_train[:,i]=labelencoder_x.fit_transform(x_train[:,i])

x_train[:,7]=labelencoder_x.fit_transform(x_train[:,7])

x_train

labelencoder_y=LabelEncoder()

y_train =labelencoder_y.fit_transform(y_train)

y_train

for i in range(0,5):
  x_test[:,i]=labelencoder_x.fit_transform(x_test[:,i])

x_test[:,7]=labelencoder_x.fit_transform(x_test[:,7])

labelencoder_y=LabelEncoder()
y_test =labelencoder_y.fit_transform(y_test)

x_test

y_test

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy', random_state=0)
dtc.fit(x_train, y_train)

y_pred=dtc.predict(x_test)

y_pred

from sklearn import metrics
print('The accuracy of decision tree: ', metrics.accuracy_score(y_pred, y_test))

from sklearn.naive_bayes import GaussianNB
nbc=GaussianNB()
nbc.fit(x_train, y_train)

y_pred=nbc.predict(x_test)

y_pred

print('The accuracy of Naive Bayes : ', metrics.accuracy_score(y_pred, y_test))

tsd = pd.read_csv("ltest.csv")

tsd.head()

tsd.info()

tsd.isnull().sum()

tsd['Gender'].fillna(tsd['Gender'].mode()[0], inplace=True)
tsd['Dependents'].fillna(tsd['Dependents'].mode()[0], inplace=True)
tsd['Self_Employed'].fillna(tsd['Self_Employed'].mode()[0], inplace=True)
tsd['Loan_Amount_Term'].fillna(tsd['Loan_Amount_Term'].mode()[0], inplace=True)
tsd['Credit_History'].fillna(tsd['Credit_History'].mode()[0], inplace=True)

tsd.isnull().sum()

tsd.boxplot(column='LoanAmount')

tsd.boxplot(column='ApplicantIncome')

tsd.LoanAmount=tsd.LoanAmount.fillna(tsd.LoanAmount.mean())

tsd['LoanAmount_log']=np.log(tsd['LoanAmount'])

tsd.isnull().sum()

tsd['TotalIncome']=tsd['ApplicantIncome']+tsd['CoapplicantIncome']
tsd['TotalIncome_log']=np.log(tsd['TotalIncome'])

tsd.head()

test=tsd.iloc[:, np.r_[1:5, 9:11, 13:15]].values

for i in range(0,5):
  test[:,i]=labelencoder_x.fit_transform(test[:,i])

test[:,7]=labelencoder_x.fit_transform(test[:,7])

test

test=ss.fit_transform(test)

pred=nbc.predict(test)

pred

