from housing_sample import *
from pandas.tools.plotting import scatter_matrix

# We're going to be digging into the training set some more. If the training
# set was very large we would sample an exploration set at this stage.
housing = strat_train_set.copy()

# Let's visualize what we're working with. This plot should look like
# California. More densly populated areas will be darker.
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()


# Now let's take a look at housing prices
# housing.plot(
#     kind="scatter",
#     x="longitude",
#     y="latitude",
#     alpha=0.4,
#     s=housing["population"] / 100, # Radius of each circle
#     label="Population",
#     c="median_house_value",        # Median house value determines color
#     cmap=plt.get_cmap("jet"),      # Predefined color map
#     colorbar=True
# )
# plt.legend()
# plt.show()

# Compute a standard correlation coefficient. The correlation coefficient
# ranges from -1 to 1. When it is close to 1, it means that there is a strong
# positive correlation. When the coefficient is close to -1, it means that
# there is a strong negative correlation. Finally, coefficients close to zero
# mean that there is no linear correlation.
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# Another way to check for correlation between attributes is to use Pandas’
# scatter matrix function which plots every numerical attribute against every
# other numerical attribute.
attributes = [
    "median_house_value",
    "median_income",
    "total_rooms",
    "housing_median_age"
]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

# Based off of the correlation coefficient and the scatter matrix, the most
# promising attribute to predict the median house value is the median
# income, so let’s zoom in on their correlation scatterplot.
# housing.plot(
#     kind="scatter",
#     x="median_income",
#     y="median_house_value",
#     alpha=0.1)
# plt.show()


# This plot reveals a few things:
#
# 1. first the correlation is indeed very strong, you can clearly see the
#    upward trend and the points are not too dispersed.
# 2. Second, the price cap that we noticed earlier is clearly visible as a
#    horizontal line at $500,000. But this plot reveals other less obvious
#    straight lines: a horizontal line around $450,000, another around $350,000,
#    perhaps one around $280,000 and a few more below that.
#
# You may want to try removing the corresponding districts to prevent your
# algorithms from learning to reproduce these data quirks.
