
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('/content/water_potability.csv')

df.sample(5)

df['Potability'].value_counts()

df.describe()

df.info()


df['ph'].fillna(np.mean(df['ph']),inplace=True)
df['Sulfate'].fillna(np.mean(df['Sulfate']),inplace=True)
df['Trihalomethanes'].fillna(np.mean(df['Trihalomethanes']),inplace=True)

df.info()

scaler=StandardScaler()
newdf=pd.DataFrame(scaler.fit_transform(df.drop(columns=['Potability'])))
newdf.head()

X=newdf.iloc[:,:-1]
y=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

clf = LogisticRegressionCV().fit(X_train, y_train)
print(clf.score(X_train,y_train))

clf.score(X_test,y_test)
