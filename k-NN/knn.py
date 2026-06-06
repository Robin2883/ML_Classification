import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#dataset loaded
df=pd.read_csv('Social_Network_Ads.csv')
df.head()

df.info()
df.describe()
df.isnull().sum()

#separating independent & dependent variables
X=df.iloc[:, :-1].values
y=df.iloc[:, -1].values

#splitting train & test data
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

#scaling features
sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.fit(X_test)
print(X_train_scaled)
print(X_test_scaled)

#training model
regr=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
regr.fit(X_train, y_train)

#prediction 
y_pred=regr.predict(X_test)
print(y_pred)

proba=regr.predict_proba(X_test)
print(proba)

accuracy=accuracy_score(y_test, y_pred)
accuracy

confusion=confusion_matrix(y_test, y_pred)
sns.heatmap(confusion, annot=True)
plt.xlabel('Prediction')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('knn_conf_matrix.png')
plt.show()

regr.predict([[27, 80000]])