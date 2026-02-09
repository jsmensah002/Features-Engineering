#Importing file using pandas
import pandas as pd
df = pd.read_csv('Melbourne_housing_FULL.csv')
print(df)

#Handling null values
print(df.isna().sum())

#Price is our target: So all rows with null price would be dropped
df =df.dropna(subset=['Price'])

#Distance and postcode have just one null value each hence can be dropped
df = df.dropna(subset=['Distance'])
df = df.dropna(subset=['Postcode'])

#The null values in Bedroom2, Bathroom, Car, Landsize, BuildingArea, and Propertycount columns would replaced with the median
cols1 = ['Bedroom2','Bathroom','Car','Landsize','BuildingArea','Propertycount']
df[cols1] = df[cols1].fillna(df[cols1].median())

#Null values in YearBuilt, CouncilArea, Lattitude, Longitude, and Regionname would be handled using forward fill (ffill) and backward fill (bfill)
col2 = ['YearBuilt','CouncilArea','Lattitude','Longtitude','Regionname']
df[col2] = df[col2].ffill().bfill()

#All null values handled

#Defining x with the top 3 features with respect to importance
regionname = df['Regionname']
regionname_dummies = pd.get_dummies(regionname,dtype=float)
address = df['Regionname']
address_dummies = pd.get_dummies(address,dtype=float)
type = df['Type']
type_dummies = pd.get_dummies(type,dtype=float)
suburb = df['Suburb']
suburb_dummies = pd.get_dummies(suburb,dtype=float)
date = df['Date']
date_dummies = pd.get_dummies(date,dtype=float)
sellerG = df['SellerG']
sellerG_dummies = pd.get_dummies(sellerG,dtype=float)

x = pd.concat([df['Rooms'],
               df['Distance'],
               address_dummies,
               type_dummies,
               regionname_dummies,
               df['Landsize'],
               df['Longtitude'],
               df['BuildingArea'],
               df['Lattitude'],
               suburb_dummies,
               date_dummies,
               sellerG_dummies],axis='columns')

y = df['Price']

#Importing prediction models
import sklearn.linear_model as lm
reg = lm.LinearRegression()
reg.fit(x,y)

from sklearn.svm import SVR
svm = SVR()
svm.fit(x,y)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)

#Importing traintest split 
from sklearn.model_selection import train_test_split as tt
x_train,x_test,y_train,y_test = tt(x,y,test_size=0.2,random_state=42)
reg.fit(x_train,y_train)
svm.fit(x_train,y_train)
rf.fit(x_train,y_train)

#Checking r^2 values for each model
print(reg.score(x,y))
print(reg.score(x_train,y_train))
print(reg.score(x_test,y_test))

print(svm.score(x,y))
print(svm.score(x_train,y_train))
print(svm.score(x_test,y_test))

print(rf.score(x,y))
print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))





