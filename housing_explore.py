import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix

# Dataset available at: https://github.com/ageron/handson-ml/tree/master/datasets/housing
HOUSING_PATH = "../datasets/housing/"

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def check_distribution(data, category):
    return data[category].value_counts() / len(data)

# Load the housing data into memory
housing = load_housing_data()

# When we want to explore the structure of a data frame we can use the
# info() method. Did you learn anything interesting about this dataset after
# exploring its structure?
# housing.info()

# How do we explore categorical attributes?
# housing["ocean_proximity"].value_counts()

# How do I get a summary of the measures of central tendency?
# housing.describe()

# How do I plot a histogram of our housing data?
# housing.hist(bins=50, figsize=(20,15))
# plt.show()


# What do you notice about these plots?
#
# 1. The data have very different scales. Some feature scaling will be required.
# 2. The median house value has been capped at 500001.0. This could pose a
#    problem because our algorithms may learn that prices never go beyond the
#    max value. If the model needs to be able to make predictions beyond the
#    max value, we could collect proper labels for each example that was
#    capped, or we could remove the example from the training and test sets.
# 3. Many histograms are skewed to the right. We need to try transforming
#    these attributes to give them more of a bell-shape to make it easier for
#    our algorithm to detect patterns.
