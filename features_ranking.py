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

#Catgeorical and numerical features
#Price column excluded because its the target variable
categorical = df[['Suburb','Address','Type','Method','SellerG','Date','CouncilArea','Regionname']]
numerical = df[['Rooms','Distance','Postcode','Bedroom2','Bathroom','Car','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude','Propertycount']]

#Creating dummy variables for the categorical columns
categorical_dummies = pd.get_dummies(categorical,dtype=float)

#Defining x and y
x = pd.concat([numerical,categorical_dummies],axis='columns')
y = df['Price']

#Importing RandomForestRegressor to rank the column importance
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(x,y)

features_importance = pd.DataFrame({
    'Features':x.columns,
    'Importance':rf.feature_importances_
})

features_importance['ParentFeatures'] = features_importance['Features'].str.split('_').str[0]
ranked_features = features_importance.groupby('ParentFeatures')['Importance'].sum().sort_values(ascending=False)
print(ranked_features)


