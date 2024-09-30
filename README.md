# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
step1:Start the program.
step2:Import numpy as np.
step3:Give the header to the data.
step4:Find the profit of population.
step5:Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
step6:End the program.
```
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: BASKAR U
RegisterNumber:  212223220013
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate = 0.1,num_iters=1000):
    x=np.c_[np.ones(len(x1)),x1]
    theta = np.zeros(x.shape[1]).reshape(-1,1)
    for _ in range(num_iters):                    
        #calculate predictions
        predictions = (x).dot(theta).reshape(-1,1)
                     
        #calculate errors
        errors = (predictions-y).reshape(-1,1)
        #update theta using gradient descent
        theta-= learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta

data = pd.read_csv("C:/Users/ANANDAN S/Documents/ML labs/50_Startups.csv")
data.head()
#Assuming the last column is your target variable y
x= (data.iloc[1:,:-2].values)
x1 = x.astype(float)
scaler = StandardScaler()
y =(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled = scaler.fit_transform(x1)
y1_scaled = scaler.fit_transform(y)
print(x)
print(x1_scaled)
#learn model parameters
theta = linear_regression(x1_scaled,y1_scaled)
#predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_scaled),theta)
prediction = prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"prediction value: {pre}")
```

## Output:
head:

![Screenshot 2024-09-23 112855](https://github.com/user-attachments/assets/da9f1210-0933-49e9-83cf-593faf2fc5ba)
![Screenshot 2024-09-23 112935](https://github.com/user-attachments/assets/17e3a463-d33d-41aa-b0e8-210dca6426ae)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
