from housing_prepare import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# NOTE
# In order to save a trained model:
#
# from sklearn.externals import joblib
# joblib.dump(my_model, "my_model.pkl")
# ...
# ...
# my_model_loaded = joblib.load("my_model.pkl")

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("STD:", scores.std())

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

print(forest_rmse)

forest_scores = cross_val_score(
    forest_reg,
    housing_prepared,
    housing_labels, scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_scores)
