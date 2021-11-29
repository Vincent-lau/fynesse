# This file contains code for supporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

import numpy as np
import geopandas as gpd
import math
import statsmodels.api as sm
import sklearn.model_selection as ms


"""Address a particular question that arises from the data"""


def fake_dist_pt(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def dist_pt(p1, p2):
    points_df = gpd.GeoDataFrame({'geometry': [p1, p2]}, crs='EPSG:4326')
    points_df = points_df.to_crs('EPSG:27700')  # UK
    # We shift the dataframe by 1 to align pnt1 with pnt2
    points_df2 = points_df.shift()
    return points_df.distance(points_df2)[1]
