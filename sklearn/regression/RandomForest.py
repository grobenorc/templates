# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:30:32 2023

@author: claes

Random Forest example with Sk-learn.

    - Hyperparameters in Random-forest (sklearn):
        # Decision trees in the forest (n_estimators),
        # Number of splits each Decision-Tree is allowed (max_depth)

    The script consists of two main parts:
    - Simple model without tuning
    - Model tuning with GridSearchCV --> Cross-validation for hyperparameter optimization
        - Only n_estimators
        - Only max_depth
        - Both simultaneously/grid_search

"""

# Importing the libraries
import numpy as np # for array operations
import pandas as pd # for working with DataFrames
import matplotlib.pyplot as plt # for data visualization

# scikit-learn modules
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.metrics import mean_squared_error # for calculating the cost function
from sklearn.ensemble import RandomForestRegressor # for building the model


# %% Loading of data
df = pd.read_csv('C:/Users/claes/OneDrive/Python/Datasets/petrol_consumption.csv')
df.columns

X = df.drop('Petrol_Consumption', axis = 1) # Features
y = df['Petrol_Consumption'] # Target

# Split the data into train/test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=7) # We use 20% as test size.


#%% Without cross-validation/tuning of parameters
model = RandomForestRegressor(n_estimators=100, # Default of sklearn
                              max_depth=None, # Default of sklearn
                              random_state=7)

# Fit model
model.fit(X_train, y_train)

# Eavluate the model

## Predict y with X_test
y_pred_no_tuning = model.predict(X_test)

## Prediction error:
mse_no_tuning = mean_squared_error(y_test, y_pred_no_tuning)
rmse_no_tuning = mean_squared_error(y_test, y_pred_no_tuning, squared=False)

print("============================================================")
print("\t\t\t\t\t\t NO TUNING")
print(f'n_estimators:\t{model.n_estimators}\nmax_depth:\t\t{model.max_depth}\nMSE:\t\t\t{mse_no_tuning:.2f}\nRMSE:\t\t\t{rmse_no_tuning:.2f}')
print("============================================================")

#%% Tuning of hyperparameters
# Because we have a small dataset we can use GridSearchCV, consider using RandomSearch for larger datasets (due to runtime).
from sklearn.model_selection import GridSearchCV

#%%% # of tree (n_estimators)

model = RandomForestRegressor(max_depth=None,
                              random_state=7) # Use the same as previous

params = {
    'n_estimators':np.arange(1, 51)
    }
    
scoring = 'neg_mean_squared_error'
n_estimator_search = GridSearchCV(model, 
                                  param_grid=params,
                                  scoring=scoring)
n_estimator_search.fit(X_train, y_train)

plt.figure(dpi=150)
plt.scatter(np.arange(1,len(params['n_estimators'])+1), n_estimator_search.cv_results_['mean_test_score'])
plt.title('Cross-validation Score')
plt.xlabel('n_estimator (#Decision Trees)')
plt.ylabel(f'{scoring}')


## Predict y with X_test
y_pred_n_estimator = n_estimator_search.predict(X_test)

## Prediction error:
mse_n_estimators = mean_squared_error(y_test, y_pred_n_estimator)
rmse_n_estimators = mean_squared_error(y_test, y_pred_n_estimator, squared=False)

print("============================================================")
print("\t\t\t\t\t\t TUNING OF n_estimators\n ")
print(f'Best # of decision trees {n_estimator_search.best_params_["n_estimators"]}.')
print(f'\nmax_depth:\t\t{model.max_depth}\nMSE:\t\t\t{mse_n_estimators:.2f}\nRMSE:\t\t\t{rmse_n_estimators:.2f}')
print("\n============================================================")



#%%% Maximum depth

model = RandomForestRegressor(n_estimators=100,
                              random_state=7)

params = {
    'max_depth': np.arange(1,51)
    }

scoring = 'neg_mean_squared_error'
max_depth_search = GridSearchCV(model, 
                                  params,
                                  scoring=scoring)

max_depth_search.fit(X_train, y_train)

plt.figure(dpi=150)
plt.scatter(np.arange(1,len(params['max_depth'])+1), max_depth_search.cv_results_['mean_test_score'])
plt.title('Cross-validation Score')
plt.xlabel('max_depth (Maximum depth of each tree)')
plt.ylabel(f'{scoring}')

## Predict y with X_test
y_pred_n_estimator = max_depth_search.predict(X_test)

## Prediction error:
mse_n_estimators = mean_squared_error(y_test, y_pred_n_estimator)
rmse_n_estimators = mean_squared_error(y_test, y_pred_n_estimator, squared=False)

print("============================================================")
print("\t\t\t\t\t\t TUNING OF max_depth\n ")
print(f'Best of max_depth {max_depth_search.best_params_["max_depth"]}.')
print(f'\nn_estimators:\t{model.n_estimators}\nMSE:\t\t\t{mse_n_estimators:.2f}\nRMSE:\t\t\t{rmse_n_estimators:.2f}')
print("\n============================================================")



#%%% Both simultaneously
from mpl_toolkits.mplot3d import Axes3D # Since

model = RandomForestRegressor(random_state=7)

params = {
    'n_estimators': np.arange(1, 51),
    'max_depth': np.arange(1, 51)
    }

scoring = 'neg_mean_squared_error'
n_estimator_max_depth_search = GridSearchCV(model, 
                                  params,
                                  scoring=scoring)

n_estimator_max_depth_search.fit(X_train, y_train)

CV_results = pd.concat([pd.DataFrame(n_estimator_max_depth_search.cv_results_['params']),
                        pd.Series(n_estimator_max_depth_search.cv_results_['mean_test_score'], name='mean_test_score')], 
                        axis=1)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(xs=CV_results['n_estimators'], 
                     ys=CV_results['max_depth'], 
                     zs=CV_results['mean_test_score'], 
                     c=CV_results['mean_test_score'], 
                     cmap='viridis')
# Adding labels and title
ax.set_xlabel('Max Depth')
ax.set_ylabel('N Estimators')
ax.set_zlabel('Mean Test Score')
ax.set_title('3D Plot of Grid Search Results')

# Adding color bar
cbar = plt.colorbar(scatter, shrink=0.6)
# cbar.set_label()
ax.view_init(10, -40) 
plt.tight_layout()
plt.show()


## Predict y with X_test
y_pred_both = n_estimator_max_depth_search.predict(X_test)

## Prediction error:
mse_both = mean_squared_error(y_test, y_pred_both)
rmse_both= mean_squared_error(y_test, y_pred_both, squared=False)

print("============================================================")
print("\t\t\t\t\t\t TUNING OF both parameters\n ")
print(f'Best parameters {n_estimator_max_depth_search.best_params_}.')
print(f'\nMSE:\t\t\t{mse_both:.2f}\nRMSE:\t\t\t{rmse_both:.2f}')
print("\n============================================================")
