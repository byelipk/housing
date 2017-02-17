from housing_prepare import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("STD:", scores.std())

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

print(lin_rmse)
# 68628.1981985

# If we look at how median house value is spread, most are between $119,800
# and $263,900. That makes value of $68,628 not very good. This is an example
# of a model underfitting the training data.
housing_labels.describe()

# When our model suffers from underfitting it can mean the features do not
# provide enough information to make good predictions, or the model is not
# powerful enough. The main ways to fix underfitting are to provide our model
# with better features, select a more powerful model, or reduce the constraints
# on the model (i.e. regularization).


# Let's compute the cross-validation scores
lin_scores = cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels, scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

# We still don't seem to be performing well.
display_scores(lin_rmse_scores)
