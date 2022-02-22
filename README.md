# Logistic-regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

incomes = pd.read_csv('diabetes.csv',na_values="?")
D = incomes.copy()
DM = D.dropna(axis=0)

# classifying the salstat as 0 or 1
#DM['Outcome'] = DM['Outcome'].map({' less than or equal to 50,000': 0, ' greater than 50,000': 1})

# separating the columns by subsetting them

col = ['Insulin','SkinThickness','Age',]
DM = DM.drop(columns=col, axis=1)

newDM = pd.get_dummies(DM, drop_first=True)

# getting the column names into a list
columnlist = list(newDM.columns)

# separating the independent values (SalStat) from dependent values
feature = list(set(columnlist) - {'Outcome'})

# getting values into y and x

y = newDM['Outcome'].values
x = newDM[feature].values

# training and testing data

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=0)

# making instance of model

log = LogisticRegression()

# fitting values of x and y

log.fit(trainx, trainy)

# prediction from trained data

pred = log.predict(testx)
#pred1 = log.predict()

# developing a confusion matrix
confuse = confusion_matrix(testy, pred)
#print(confuse)

accuracy = accuracy_score(testy, pred)
acc='%.2f'%accuracy
ac=(float(acc))
#print("The accuracy of the model is :",'%.2f'%accuracy)
print("The accuracy of the model is:",ac*100,"%")
print("Misclassified samples : %d" % (testy!=pred).sum())
