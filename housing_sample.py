from housing_explore import *
from sklearn.model_selection import StratifiedShuffleSplit

def check_distribution(data, category):
    return data[category].value_counts() / len(data)


# Stratified Sampling
#
# Create a training and test set using stratified sampling methods.
#
# Our dataset should be representative of the population we're trying to
# generalize about. The same holds true for our data splits. Suppose median
# income was a very important attribute to predict median housing prices. We
# would need to ensure our training and test set is representative of the
# various categories of median income in the whole dataset.
#
# If you look at the histogram for median income again, you'll notice that
# the majority of values hover around the 2-6 range. But some values are much
# larger than 6. We need to ensure we have a sufficient number of instances in
# each dataset for each stratum.
# housing["median_income"].hist()
# plt.show()

# Create an "income_cat" attribute that ranges from 1 to 5.
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)

# Some values will exceed 5. So let's merge those back into category 5.
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# Now we can do stratified sampling based on the "income_cat" attribute
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_idx]
    strat_test_set  = housing.loc[test_idx]


# Now check the distributions in our data. Each data set should have a
# similar distribution.
#
# check_distribution(housing, "income_cat")
# check_distribution(strat_train_set, "income_cat")
# check_distribution(strat_test_set, "income_cat")


# Since we've suceeded in sampling the data properly it's ok to remove
# "income_cat" so the housing data is back to its original state.
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)
