# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VESLIN ANISH A
RegisterNumber: 212223240175
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SMARTLINK/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![Screenshot 2024-08-30 225433](https://github.com/user-attachments/assets/df1558eb-63f7-42ff-94a6-5b1ce3c0d6ea)
![Screenshot 2024-08-30 225458](https://github.com/user-attachments/assets/16d328f9-7dcc-44f5-86b9-d00942ed78d2)
![Screenshot 2024-08-30 225532](https://github.com/user-attachments/assets/15b609ba-aa78-48f2-9a64-7d70efbd529b)
![Screenshot 2024-08-30 225553](https://github.com/user-attachments/assets/f42cde42-0655-41b0-b7ab-34bb977bb29a)
![Screenshot 2024-08-30 225616](https://github.com/user-attachments/assets/111d7436-499c-4bbf-92f7-75bc8702112f)
![Screenshot 2024-08-30 225633](https://github.com/user-attachments/assets/18e88c4e-24bf-4578-87d5-8976396f1562)
![Screenshot 2024-08-30 225719](https://github.com/user-attachments/assets/5fb791e4-2552-46da-bc14-87a3aee61f7f)
![Screenshot 2024-08-30 225736](https://github.com/user-attachments/assets/74b834e8-a7fd-4398-8881-e35f3173271c)
![Screenshot 2024-08-30 225753](https://github.com/user-attachments/assets/191351bf-e271-4492-95f6-7497b8e12cd4)
![Screenshot 2024-08-30 225815](https://github.com/user-attachments/assets/2ef9b6a2-7dee-4a10-a51c-901452612e9f)
![Screenshot 2024-08-30 225838](https://github.com/user-attachments/assets/529ff4d7-5117-438a-8457-acee8b6c7b79)




















## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
