from housing_prepare import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("STD:", scores.std())

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

print(tree_rmse)
# 0.0

# No error at all? We've probably overfit the data. But how do we know? We can
# create a validation set to find out.


# The following code performs K-fold cross-validation: it randomly splits the
# training set into 10 distinct subsets called folds, then it trains and
# evaluates the decision tree model 10 times, picking a different fold for
# evaluation every time and training on the other 9 folds.
tree_scores = cross_val_score(
    tree_reg,
    housing_prepared,
    housing_labels, scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-tree_scores)

# We still don't seem to be performing well.
display_scores(tree_rmse_scores)
