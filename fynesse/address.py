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

from abc import get_cache_token
from matplotlib import pyplot as plt
import numpy as np
import geopandas as gpd
import math
import statsmodels.api as sm
import sklearn.model_selection as ms
import sklearn.metrics
import pandas as pd
from . import access
from . import assess

"""Address a particular question that arises from the data"""


def fake_dist_pt(p1, p2):
    """
    this function is much faster than the one below although it does not compute
    the actual distance
    """
    scaler = 10**6
    p2 = (float(p2[0]), float(p2[1]))
    d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return d * scaler


def dist_pt(p1, p2):
    points_df = gpd.GeoDataFrame({'geometry': [p1, p2]}, crs='EPSG:4326')
    points_df = points_df.to_crs('EPSG:27700')  # UK
    # We shift the dataframe by 1 to align pnt1 with pnt2
    points_df2 = points_df.shift()
    return points_df.distance(points_df2)[1]


def get_closet_dist_to_facility(house_loc, pois, fac_name):
    house_to_place = []
    for h in house_loc.loc[:, ['longitude', 'latitude']].values:

        for t in assess.get_tags():
            ame = pois[pois[t] == fac_name]
            if (ame.shape[0] == 0):
                continue

            dist = float('inf')

            if 'way' in ame.index:
                for a in ame["geometry"]['way']:
                    p1 = (a.centroid.x, a.centroid.y)
                    p2 = (h[0], h[1])

                    dist = min(fake_dist_pt(p1, p2), dist)

            if 'node' in ame.index:
                for a in ame['geometry']['node']:
                    p1 = (a.x, a.y)
                    p2 = (h[0], h[1])

                    dist = min(fake_dist_pt(p1, p2), dist)

            house_to_place.append(dist)

    return house_to_place


def design_matrix(x, constant):
    design = np.empty((0, x.shape[1]), int)
    for z in x:
        design = np.append(design, np.array([z, z**2]), axis=0)
    design = np.concatenate((design, constant))
    return np.column_stack(design)


def get_house_dist_to_facilities(house_loc, pois):
    """
    This returns a map that maps each place/amenity to a list of distances.
    This list of distances are the shortest distance of each place to each house
    in house_loc
    :param house_loc: a dataframe that is the result of the join on pp_data and 
    postcode_data
    :param pois: This contains all the facility/amenities that we are interested 
    in measuring distances
    """

    facility_names = get_all_facility_names(pois)

    house_dist_to_places = {fac_name: get_closet_dist_to_facility(house_loc, pois, fac_name)
                            for fac_name in facility_names}

    return house_dist_to_places


def get_all_facility_names(pois):
    l = np.array([name for osm_tag in assess.get_tags()
                 for name in pois[osm_tag].unique() if pd.notnull(name)])
    return l


def feature_selection(house_loc, pois, draw=False):
    """
    returns a list of keys that can be used as features
    here features would be represented as the shortest dist from the house to that
    particular facility/place

    select features by plotting graphs and finding corr between price and the shortest
    dist

    """

    house_dist_to_places = get_house_dist_to_facilities(house_loc, pois)
    facility_names = get_all_facility_names(pois)

    corrs = []

    r = int(len(facility_names) / 2)
    if draw:
        fig, ax = plt.subplots(r, 2)
        fig.set_size_inches(13, 10)
    for i in range(r):
        for j in range(2):
            name = facility_names[i*2+j]
            y = house_loc['price'].values
            x = house_dist_to_places[name]
            c = np.corrcoef(x, y)[0, 1]
            corrs.append((c, name))
            if draw:
                ax[i, j].scatter(x, y)
                ax[i, j].set_title(facility_names[i * 2 + j])
    corrs = sorted(corrs, key=lambda c: abs(c[0]), reverse=True)

    feature_keys = []
    if len(corrs) >= 3:
        feature_keys = [k[1] for k in corrs[:3]]

    if draw:
        print(f"fetched facilities are {facility_names}")
        print(corrs)
        fig.tight_layout()
        plt.show()
    return feature_keys


def get_features_selected(house_dist_to_places, pois, feature_keys):
    """
    Given facilities we are interested in, select all those from house_dist_to_places
    :param house_dist_to_places: a dictionary mapping amenities to shortest distance
    :param pois: has all the facilities/amenities that we might need
    """

    res = []
    for k in feature_keys:
        if k in house_dist_to_places:
            res.append(house_dist_to_places[k])
        else:
            assert(False)
    return np.array(res)


def train(house_loc_train, pois, property_type, feature_keys, fig, ax):
    print("====================training=====================")
    house_dist_to_places = get_house_dist_to_facilities(house_loc_train, pois)

    x = get_features_selected(house_dist_to_places, pois, feature_keys)
    y = np.array(house_loc_train['price'].values)

    prop_types = ['F', 'D', 'S', 'O', 'T']
    properties = house_loc_train['property_type']

    i = [np.where(properties == k, 1, 0) for k in prop_types]

    # modelling with design matrix and fit the model
    design = design_matrix(x, i)

    m_linear_basis = sm.OLS(y, design)
    results_basis = m_linear_basis.fit()

    x0_pred = np.array(np.linspace(np.min(x[0]), np.max(x[0]), 500))
    x1_pred = np.array(np.linspace(np.min(x[1]), np.max(x[1]), 500))
    x2_pred = np.array(np.linspace(np.min(x[2]), np.max(x[2]), 500))
    x_pred = np.array([x0_pred, x1_pred, x2_pred])

    prop_pred = np.array([property_type for _ in range(500)])

    i_pred = [np.where(prop_pred == k, 1, 0) for k in prop_types]
    design_pred = design_matrix(x_pred, i_pred)
    y_pred_linear_basis = results_basis.get_prediction(
        design_pred).summary_frame(alpha=0.05)

    ax.scatter(x[0], y, zorder=2)

    ax.plot(x_pred[0], y_pred_linear_basis['mean'], color='red',
            zorder=1)

    return results_basis


def draw_test_data(house_loc_test, pois, feature_keys, fig, ax):
    facility_names = get_all_facility_names(pois)

    house_dist_to_places = get_house_dist_to_facilities(house_loc_test, pois)
    x = get_features_selected(house_dist_to_places, pois, feature_keys)

    y = np.array(house_loc_test['price'].values)
    ax.scatter(x[0], y, zorder=2)

    plt.show()


def predict_price(latitude, longitude, date, property_type, draw=False):

    # get the prices_coordinate_data
    house_loc = assess.retrieve_houses(
        access.get_conn(), latitude, longitude, date, property_type)

    # get OSM data
    pois = assess.retrieve_pois(latitude, longitude, house_loc)

    if (draw):
        # draw the date before we fit the model
        place_name = assess.get_place_name(house_loc)
        assess.draw_location(access.get_conn(), latitude,
                             longitude, date, place_name)

    fig, ax = plt.subplots()
    fig.set_size_inches(13, 10)
    # train and validate

    house_loc_train, house_loc_test = ms.train_test_split(
        house_loc, random_state=0)

    feature_keys = feature_selection(house_loc, pois)
    results = train(house_loc_train, pois,
                    property_type, feature_keys, fig, ax)
    draw_test_data(house_loc_test, pois, feature_keys, fig, ax)
    validate(results, house_loc_test, pois, feature_keys)

    # get the final prediction
    r = predict_single(results, house_loc_test, pois,
                       longitude, latitude, property_type, feature_keys)
    return r


def predict_single(results, house_loc_test, pois, longitude, latitude,
                   property_type, feature_keys):

    house_loc_test.loc[-1, 'latitude'] = latitude
    house_loc_test.loc[-1, 'longitude'] = longitude
    house_loc_test = house_loc_test.tail(1)

    house_dist_to_places = get_house_dist_to_facilities(house_loc_test, pois)

    x = get_features_selected(house_dist_to_places, pois, feature_keys)

    prop_types = assess.get_prop_types()
    properties = np.array([property_type])
    i_pred = [np.where(properties == k, 1, 0) for k in prop_types]

    design_pred = design_matrix(x, i_pred)

    y_pred = results.get_prediction(design_pred).summary_frame(alpha=0.05)

    return y_pred['mean'][0]


def validate(results, house_loc_test, pois, feature_keys):
    print("==========validating on test dataset==============")

    house_dist_to_places = get_house_dist_to_facilities(
        house_loc_test, pois)

    x = get_features_selected(house_dist_to_places, pois, feature_keys)
    y = np.array(house_loc_test['price'].values)

    prop_types = assess.get_prop_types()
    properties = house_loc_test['property_type']
    i_pred = [np.where(properties == k, 1, 0) for k in prop_types]

    design_pred = design_matrix(x, i_pred)

    y_pred = results.get_prediction(design_pred).summary_frame(alpha=0.05)

    err = sklearn.metrics.mean_squared_error(y, y_pred['mean'].values)
    print(f"the square root of mean squared error is {math.sqrt(err)}")

    fig, ax = plt.subplots()
    ax.scatter(y, y_pred['mean'])

    ax.set_xlim([0, np.max(y)])
    ax.set_ylim([0, np.max(y_pred['mean'])])
    ax.set_title("Actual price against predicted price")
    ax.set_xlabel('predicted prices')
    ax.set_ylabel('actual prices')

    # y = x
    x = np.linspace(0, np.max(y), 1000)
    y = x
    ax.plot(x, y, color='magenta')

    return y_pred['mean']
