import pandas as pd
import numpy as np

path = "diabetes.csv"
diab = pd.read_csv(path)

diab.shape
diab.columns
diab.describe()
diab.isnull().sum()

x = diab.values[:,0:8]
x

y = diab.values[:,8]
y

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x ,y, test_size=0.35, random_state=1)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train,y_train)

pdct = logreg.predict(x_test)
pdct

#actual = list(y_test)


#df_results = pd.DataFrame({'actual':actual, 'predicted':pdct})
#print(df_results)

#method 1 for accuracy
from sklearn.metrics import classification_report
classification_report(y_test,pdct)

#method 2 for accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pdct)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,pdct)
