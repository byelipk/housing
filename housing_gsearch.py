from housing_prepare import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# See: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2,4,6,8]},
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4] }
]

# The param grid tells scikit-learn to first evaluate all 3x4 combinations
# of the 'n_estimators' and 'max_features' hyperparameters. Then it
# evaluates all the 2x3 combinations with the 'bootstrap' hyperparameter set
# to False.

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(
    forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")

grid_search.fit(housing_prepared, housing_labels)

# Find the best set of hyperparameters
grid_search.best_params_

# Find the best model
grid_search.best_estimator_

# See the evaluation scores for each model
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
