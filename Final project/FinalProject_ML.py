#!/usr/bin/env python
# coding: utf-8

# In[416]:


import pandas as pd
import numpy as np
import re as re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import sys
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.preprocessing import scale 
import seaborn as sns
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn import tree # for decision tree models

import plotly.express as px  # for data visualization
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import graphviz

print(sys.path)
import xlsxwriter


# ### Data Load ################

# In[417]:


SD_AirbnbData = pd.read_csv('SanDiego_2019.csv')
#print(NY_AirbnbData)

print(SD_AirbnbData.describe())


# In[418]:


#print(data.head(5))
print(SD_AirbnbData.info())
print(SD_AirbnbData.shape[0])


# In[419]:


print(SD_AirbnbData.iloc[: , 35:40].head(10))


# ### Preprocessing 
# 
# 1)Filtering only relevant columns which can be used for analysis. Removed columns such as host_id, Host_name, Summary etc.<br>
# 2)Remove $ sign from price and convert it into float <br>
# 3) Removed null values by filling in mean values for numerical columns and mode values for categorical data <br>
# 4) Checking for outlier records and removing those

# In[420]:


#A = ['neighbourhood_cleansed', 'price_per_stay', 'host_is_superhost','accommodates','beds','host_response_time','host_total_listings_count','instant_bookable','review_scores_value' ,'room_type','latitude','longitude','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count']
A = ['neighbourhood_cleansed', 'price_per_stay', 'host_is_superhost','accommodates','beds','host_response_time','host_total_listings_count','instant_bookable','review_scores_value' ,'room_type','latitude','longitude','number_of_reviews','minimum_nights','maximum_nights','number_of_stays','cancellation_policy','square_feet','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','requires_license','is_business_travel_ready','require_guest_profile_picture','guests_included','require_guest_phone_verification','host_since','host_has_profile_pic','host_identity_verified','bathrooms','bedrooms']

#'security_deposit','cleaning_fee'
#B = ['minimum_nights','maximum_nights','number_of_stays','cancellation_policy','square_feet','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','requires_license','is_business_travel_ready','require_guest_profile_picture','require_guest_phone_verification']
#C = ['host_since','host_has_profile_pic','host_identity_verified','bathrooms','bedrooms']
# Amenities


# In[421]:


RelevantInfo = SD_AirbnbData[A]
print(RelevantInfo.head())


# In[422]:


RelevantInfo["price_per_stay"] = RelevantInfo["price_per_stay"].str.replace('\$', '', regex=True)
RelevantInfo["price_per_stay"] = RelevantInfo["price_per_stay"].str.replace('\,', '', regex=True)
RelevantInfo["price_per_stay"] = RelevantInfo["price_per_stay"].apply( lambda x: float(x) if (np.all(pd.notnull(x))) else x)

#RelevantInfo["security_deposit"] = RelevantInfo["security_deposit"].str.replace('\$', '', regex=True)
#RelevantInfo["security_deposit"] = RelevantInfo["security_deposit"].str.replace('\,', '', regex=True)
#RelevantInfo["security_deposit"] = RelevantInfo["security_deposit"].apply( lambda x: float(x) if (np.all(pd.notnull(x))) else x)

#RelevantInfo["cleaning_fee"] = RelevantInfo["cleaning_fee"].str.replace('\$', '', regex=True)
#RelevantInfo["cleaning_fee"] = RelevantInfo["cleaning_fee"].str.replace('\,', '', regex=True)
#RelevantInfo["cleaning_fee"] = RelevantInfo["cleaning_fee"].apply( lambda x: float(x) if (np.all(pd.notnull(x))) else x)


# In[423]:


print(RelevantInfo.head())
print(RelevantInfo.isna().sum().sort_values())


# In[424]:


RelevantInfo.fillna({'reviews_per_month':0}, inplace=True)
#'cleaning_fee','security_deposit'

#
Number_Colms = ['bathrooms','bedrooms','square_feet','host_total_listings_count','beds','review_scores_value','review_scores_cleanliness','review_scores_communication','review_scores_accuracy','review_scores_checkin','review_scores_location']
print(RelevantInfo[Number_Colms].head())
RelevantInfo[Number_Colms]=RelevantInfo[Number_Colms].fillna(RelevantInfo.median().iloc[0])

cols_string=["host_is_superhost", "host_response_time",'host_has_profile_pic','host_identity_verified']
RelevantInfo[cols_string]=RelevantInfo[cols_string].fillna(RelevantInfo.mode().iloc[0])
RelevantInfo[cols_string]=RelevantInfo[cols_string].astype(str)

RelevantInfo.isnull().sum()


# In[425]:


## Checking for outliers in all numerical columns and removing those records

#DF_Num = RelevantInfo[varlist]

list2 = Number_Colms = ['price_per_stay','minimum_nights' ,'bathrooms','bedrooms','square_feet','host_total_listings_count','beds','review_scores_value','review_scores_cleanliness','review_scores_communication','review_scores_accuracy','review_scores_checkin','review_scores_location']

Q1 = RelevantInfo[list2].quantile(0.25)
Q3 = RelevantInfo[list2].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

UpdatedRelevantInfo = RelevantInfo[~((RelevantInfo[list2] < (Q1 - 1.5 * IQR)) |(RelevantInfo[list2] > (Q3 + 1.5 * IQR))).any(axis=1)]


# ### EDA ###

# In[426]:


corr = RelevantInfo.corr(method='kendall')
plt.figure(figsize=(15,15))
sns.heatmap(corr,annot = True)


# In[427]:


map=RelevantInfo.plot(kind='scatter', x='longitude', y='latitude', label='price_per_stay', c='price_per_stay',
                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.3, figsize=(15,15))
map.legend()


# In[428]:


print(RelevantInfo.neighbourhood_cleansed.unique())
print(len(RelevantInfo.neighbourhood_cleansed.unique()))


# In[429]:


plt.figure(figsize = (20,20))
sns.scatterplot(x=RelevantInfo.longitude,y=RelevantInfo.latitude,hue= RelevantInfo.neighbourhood_cleansed)
plt.show()


# In[430]:


print(RelevantInfo.room_type.unique())


# In[431]:


plt.figure(figsize = (20,20))
sns.scatterplot(x=RelevantInfo.longitude,y=RelevantInfo.latitude,hue= RelevantInfo.room_type)
plt.show()


# In[432]:


RelevantInfo.hist(figsize=(30,45),layout=(10,5))


# In[433]:


sns.relplot(x="price_per_stay", y="number_of_reviews", data=UpdatedRelevantInfo)


# In[434]:


sns.relplot(x="price_per_stay", y="minimum_nights", data=UpdatedRelevantInfo)


# In[435]:


sns.relplot(x="price_per_stay", y="bedrooms", data=UpdatedRelevantInfo)

## There are few outlier values which are present and have to be removed.


# ### Modelling ###

# In[436]:


SD_airbnb = pd.get_dummies(RelevantInfo, columns=['host_has_profile_pic','host_identity_verified','is_business_travel_ready','require_guest_phone_verification','require_guest_profile_picture','requires_license','cancellation_policy','neighbourhood_cleansed', 'instant_bookable','room_type',"host_is_superhost", "host_response_time"],drop_first=True)

#A = ['latitude','longitude','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','calculated_host_listings_count']

X = SD_airbnb.drop(['price_per_stay','host_since'], axis=1)
Y = SD_airbnb['price_per_stay']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#### Linear Regression #################
m1 = LinearRegression()
m1.fit(X_train, Y_train)

### Training Accuracy #########
print("Training Results")
print(explained_variance_score(Y_train, m1.predict(X_train)))
Y_pred_train = m1.predict(X_train)
mse = mean_squared_error(Y_train, Y_pred_train)
rmse = np.sqrt(mse)

print(mse)
print(rmse)
#print(m1.score(X_test,Y_test))



print("Test Results")

print(explained_variance_score(Y_test, m1.predict(X_test)))
print(m1.score(X_test,Y_test))

Y_pred = m1.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print(mse)
print(rmse)


# In[437]:


############## Linear Regression With Intercept ##################

X_train2 = sm.add_constant(X_train)
m2 = sm.OLS(Y_train, X_train2)
results = m2.fit()

print(results.summary())


# In[438]:


# Filter out the variables that have p-values < 0.05

sig_vars = results.pvalues[1:].index[results.pvalues[1:] < 0.05]
X_sig = X_train[sig_vars]
print(sig_vars)


# In[439]:


############## Lasso Model Selection via CV #################

from sklearn.linear_model import LassoCV

# Lasso with 5 fold cross-validation
modelCVLasso = LassoCV(cv=5, random_state=0, max_iter=10000)

# Fit model
modelCVLasso.fit(scale(X_train), scale(Y_train))
modelCVLasso.alpha_



# In[440]:


plt.semilogx(modelCVLasso.alphas_, modelCVLasso.mse_path_, ":")
plt.plot(
    modelCVLasso.alphas_ ,
    modelCVLasso.mse_path_.mean(axis=-1),
    "k",
    label="Average across the folds",
    linewidth=2,
)
plt.axvline(
    modelCVLasso.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
)

plt.legend()
plt.xlabel("alphas")
plt.ylabel("Mean square error")
plt.title("Mean square error on each fold")
plt.axis("tight")

("")


# In[441]:


### Lasso Model Implementation  based on above mentioned CV ##########

## Alpha ~ 0.01 as seen above

m3 = Lasso(alpha= 0.01)
m3.fit(scale(X_train), scale(Y_train))


#### Training Results #################

y_pred3 = m3.predict(X_train)

mse3 = mean_squared_error(Y_train, y_pred3)
rmse3 = np.sqrt(mse3)
print(mse3)
print(rmse3)
print(m3.score(scale(X_train),scale(Y_train)))


### TEst Results
y_pred3 = m3.predict(X_test)

mse3 = mean_squared_error(Y_test, y_pred3)
rmse3 = np.sqrt(mse3)
print(mse3)
print(rmse3)
print(m3.score(scale(X_test),scale(Y_test)))




# In[442]:


A = X_train.iloc[:,m3.coef_ != 0].columns.tolist()
print(len(A))

## 58 parameters have non zero values. Lasso shrunk half of the predictors

import eli5
eli5.show_weights(m3, top=-1, feature_names = X_train.columns.tolist())


# In[443]:


########## Decision Trees ######################

m5 = DecisionTreeRegressor(random_state=42)
m5.fit(X_train, Y_train)



#### Test Results
y_pred5 = m5.predict(X_test)
mse5 = mean_squared_error(Y_test, y_pred5)
print(np.sqrt(mse5))
print(mse5)
print(r2_score(Y_test, y_pred5))


Tree_FeatureImp = m5.feature_importances_
indices = np.argsort(m5.feature_importances_)[::-1]
print("Feature Ranking")

Importance = []
Feature_Name = []

for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f+1,X_train.columns[indices[f]],Tree_FeatureImp[indices[f]]))
    
    if f < 15:
        Feature_Name.append(X_train.columns[indices[f]])
        Importance.append(Tree_FeatureImp[indices[f]])
    

    ### Top 15 features
plt.barh(Feature_Name, Importance)




# In[444]:


######### Random Forest ###################

m6 = RandomForestRegressor(n_estimators=100, random_state=42)
m6.fit(X_train, Y_train)


### Test Results ############
y_pred6 = m6.predict(X_test)
mse6 = mean_squared_error(Y_test, y_pred6)

print(r2_score(Y_test, y_pred6))
print(np.sqrt(mse6))


##### Feature Importance ##############

#print(m6.feature_importances_)
RF_FeatureImp = m6.feature_importances_
indices = np.argsort(m6.feature_importances_)[::-1]
print("Feature Ranking")


Importance = []
Feature_Name = []

for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f+1,X_train.columns[indices[f]],RF_FeatureImp[indices[f]]))
    
    if f < 15:
        Feature_Name.append(X_train.columns[indices[f]])
        Importance.append(RF_FeatureImp[indices[f]])
    
    
plt.barh(Feature_Name, Importance)


# In[445]:


print(Y_test)

m7 = XGBRegressor()

m7.fit(X_train, Y_train, verbose=False)


y_pred7 = m7.predict(X_test)
mse7 = mean_squared_error(Y_test, y_pred7)
print(r2_score(Y_test, y_pred7))
print(np.sqrt(mse7))


# In[446]:


####### Principal Component Analysis ################


pca = PCA()
print(X.shape[1])
X_reduced = pca.fit_transform(scale(X_train))
comp = pca.fit(scale(X_train))

print(np.cumsum(pca.explained_variance_ratio_))

x = np.arange(1, 130)
f = pd.DataFrame(np.cumsum(comp.explained_variance_ratio_))[0].iloc[x]
g = 0*x + 0.75

plt.plot(x, f, '-')
plt.plot(x, g, '--',color = "brown")
plt.grid()
idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
plt.plot(x[idx], f[idx], 'ro')

plt.axvline(x = idx,linestyle='dashed',color = "brown")

plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
sns.despine()
plt.show()
#np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


# In[447]:


from sklearn import model_selection

n = len(X_reduced)
kf_10 = model_selection.KFold( n_splits=10, shuffle=True, random_state=1)

regr = LinearRegression()
mse = []
#Accuracy = []

# Calculate MSE with only the intercept (no principal components in regression)
score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), Y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using CV for the 19 principle components, adding one component at the time.
for i in np.arange(1,90):
    #print(i)
    score = -1*model_selection.cross_val_score(regr, X_reduced[:,:i], Y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(score)
    
# Plot results    
plt.plot(mse, '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('Price')
plt.xlim(xmin=-1);


# In[448]:


### Based on the elbow rule we can chose principal components to be 25 ###########


# In[449]:


pca = PCA(n_components=25).fit(X_train)
X_train_transform = pca.transform(scale(X_train))
X_train_transform.shape


# In[450]:


X_test_transform = pca.transform(scale(X_test))
X_test_transform.shape


# ### Running Lasso,, Decision Tree, Random Forest and XGBoost with Transformed X variables after PCA (ncomponents = 25)

# In[451]:


PC_m1 = LinearRegression()
PC_m1.fit(X_train_transform, (Y_train))

print(PC_m1.score(X_test_transform,Y_test))
# print(m1.intercept_)
# print(m1.coef_)
print(explained_variance_score(Y_test, PC_m1.predict(X_test_transform))) 

#### Test Results ##########
Y_pred = PC_m1.predict(X_test_transform)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print(mse)
print(rmse)


# In[452]:


############ Random Forest with PCA Components ##############

m6 = RandomForestRegressor(n_estimators=100, random_state=42)
m6.fit(X_train_transform, Y_train)

#### Test Results ##########

y_pred6 = m6.predict(X_test_transform)
mse6 = mean_squared_error(Y_test, y_pred6)
print(r2_score(Y_test, y_pred6))
print(np.sqrt(mse6))


# In[453]:


m5 = DecisionTreeRegressor(random_state=42)
m5.fit(X_train_transform, Y_train)

y_pred5 = m5.predict(X_test_transform)
mse5 = mean_squared_error(Y_test, y_pred5)
print(mse5)
print(np.sqrt(mse5))
print(r2_score(Y_test, y_pred5))


# In[454]:


m7 = XGBRegressor()

m7.fit(X_train_transform, Y_train, verbose=False)


y_pred7 = m7.predict(X_test_transform)
mse7 = mean_squared_error(Y_test, y_pred7)
print(r2_score(Y_test, y_pred7))
print(mse7)
print(np.sqrt(mse7))

