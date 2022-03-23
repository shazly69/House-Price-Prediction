import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics

# 1st Model using X: House Age
data = pd.read_csv(r"C:\Users\yousu\Desktop\Machine Learning\assignment\assignment1_dataset.csv")
print(data.describe())
x = data["house age"]
y1 = data["house price of unit area"]

l = 0.001            v      #learning rate
m = 0                       #slope
c = 0                       #y-intercept
n = float(len(data))        #number of records in data
ypred=0                     #Prediciton
epochs = 10000                
for i in range(epochs):
    ypred = m * x + c
    dm = 1 / n * sum((ypred - y1) * x)
    dc = 1 / n * sum(ypred - y1)
    m = m - l * dm
    c = c - l * dc
print('Mean Square Error', metrics.mean_squared_error(y1, ypred))

Pred = m * x + c

#plotting
plt.scatter(x, y1)
plt.xlabel('houseage', fontsize=20)
plt.ylabel('house price of unit area', fontsize=20)
plt.plot(x, Pred, color='red', linewidth=3)
plt.show()

# Predict House Price of unit area based on House Age
houseage = float(input('Enter your houseage : '))
y_test = m * houseage + c
print('Your predicted PRICE is ' + str(float(y_test)))


##############################################
# 2nd Model using X:distance to the nearest MRT station

x1 = data["distance to the nearest MRT station"]

##### standardizing the data using the formula: (x - mean)/Standard_deviation
x1 = (x1 - 1083.885689) / 1260.584387

l1 = 0.01
m1 = 0
c1 = 0
n1 = float(len(x1))
epochs1 = 1000
for i in range(epochs1):
    ypred = m1 * x1 + c1
    dm = 1 / n1 * sum((ypred - y1) * x1)
    dc = 1 / n1 * sum(ypred - y1)
    m1 = m1 - l1 * dm
    c1 = c1 - l1 * dc
print('Mean Square Error', metrics.mean_squared_error(y1, ypred))

Pred1 = m1 * x1 + c1

#### returning the data to its original format before standardizing for plotting with the right values
x1 = (x1 * 1260.584387) + 1083.885689

plt.scatter(x1, y1)
plt.xlabel('distance to the nearest MRT station', fontsize=20)
plt.ylabel('house price of unit area', fontsize=20)
plt.plot(x1, Pred1, color='red', linewidth=3)
plt.show()

# Predict House Price of unit area based on distance to the nearest MRT station
prd1 = float(input('Enter your distance to the nearest MRT station : '))
prd1 = (prd1 - 1083.886) / 1260.584
y_test1 = m1 * prd1 + c1
print('Your predicted PRICE is ' + str(float(y_test1)))


##############################################
# 3rd Model using X:number of convenience stores


x1 = data["number of convenience stores"]

l1 = 0.01
m1 = 0
c1 = 0
n1 = float(len(x1))
epochs1 = 1000
for i in range(epochs1):
    ypred = m1 * x1 + c1
    dm = 1 / n1 * sum((ypred - y1) * x1)
    dc = 1 / n1 * sum(ypred - y1)
    m1 = m1 - l1 * dm
    c1 = c1 - l1 * dc
print('Mean Square Error', metrics.mean_squared_error(y1, ypred))

Pred1 = (m1 * x1) + c1
plt.scatter(x1, y1)
plt.xlabel('number of convenience stores', fontsize=20)
plt.ylabel('house price of unit area', fontsize=20)
plt.plot(x1, Pred1, color='red', linewidth=3)
plt.show()

# Predict House Price of unit area based on number of convenience stores
prd1 = float(input('Enter number of convenience stores : '))
y_test1 = m1 * prd1 + c1
print('Your predicted PRICE is ' + str(float(y_test1)))


##############################################
# 4th Model using X:latitude


x1 = data["latitude"]

l1 = 0.00001
m1 = 0
c1 = 0
n1 = float(len(x1))
epochs1 = 1000

for i in range(epochs1):
    ypred = m1 * x1 + c1
    dm = 1 / n1 * sum((ypred - y1) * x1)
    dc = 1 / n1 * sum(ypred - y1)
    m1 = m1 - l1 * dm
    c1 = c1 - l1 * dc
print('Mean Square Error', metrics.mean_squared_error(y1, ypred))

Pred1 = (m1 * x1) + c1
plt.scatter(x1, y1)
plt.xlabel('latitude', fontsize=20)
plt.ylabel('house price of unit area', fontsize=20)
plt.plot(x1, Pred1, color='red', linewidth=3)
plt.show()

# Predict House Price of unit area based on latitude
prd1 = float(input('Enter latitude : '))
y_test1 = m1 * prd1 + c1
print('Your predicted PRICE is ' + str(float(y_test1)))


##############################################
# 5th Model using X:longitude


x1 = data["longitude"]

l1 = 0.000001
m1 = 0
c1 = 0
n1 = int(len(y1))
epochs1 = 1000

for i in range(epochs1):
    ypred = m1 * x1 + c1
    dm = 1 / n1 * sum((ypred - y1) * x1)
    dc = 1 / n1 * sum(ypred - y1)
    m1 = m1 - l1 * dm
    c1 = c1 - l1 * dc
print('Mean Square Error', metrics.mean_squared_error(y1, ypred))
Pred1 = (m1 * x1) + c1
plt.scatter(x1, y1)
plt.xlabel('longitude', fontsize=20)
plt.ylabel('house price of unit area', fontsize=20)
plt.plot(x1, Pred1, color='red', linewidth=3)
plt.show()

# Predict House Price of unit area based on latitude
prd1 = float(input('Enter longitude : '))
y_test1 = m1 * prd1 + c1
print('Your predicted PRICE is ' + str(float(y_test1)))

##############################################
# 6th Model using X:transaction date

x1 = data["transaction date"]

# taking the first 4 characters from the strings in records (year of transaction)
# discarding the dash(-) and number following it
x1=(x1.apply(lambda x:x[0:4]))
# changing the datatype of the column from string to float for calculations
x1=x1.astype(float)



l1 = 0.000000001
m1 = 0
c1 = 0
n1 = int(len(y1))
epochs1 = 1000

for i in range(epochs1):
    ypred = m1 * x1 + c1
    dm = 1 / n1 * sum((ypred - y1) * x1)
    dc = 1 / n1 * sum(ypred - y1)
    m1 = m1 - l1 * dm
    c1 = c1 - l1 * dc
print('Mean Square Error', metrics.mean_squared_error(y1, ypred))

Pred1 = (m1 * x1) + c1
plt.scatter(x1, y1)
plt.xlabel('transaction date', fontsize=20)
plt.ylabel('house price of unit area', fontsize=20)
plt.plot(x1, Pred1, color='red', linewidth=3)
plt.show()

# Predict House Price of unit area based on latitude
prd1 = float(input('Enter transaction date : '))
y_test1 = m1 * prd1 + c1
print('Your predicted PRICE is ' + str(float(y_test1)))

