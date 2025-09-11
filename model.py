import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('titanic.csv')


x= df.drop(['Name','Sex','Ticket','Age','Cabin','Embarked'],axis=1)
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)

concat=pd.concat([x_test,y_test],axis=1)
print(concat)

mse=mean_squared_error(y_pred,y_test)
r2=r2_score(y_pred,y_test)
print(mse)
print(r2)


