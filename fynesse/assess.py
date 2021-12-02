import osmnx as ox
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import *

from . import access
from . import address

"""These are the types of import we might expect in this file
import pandas
import bokeh
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""
Place commands in this file to assess the data you have downloaded. How are 
missing values encoded, how are outliers encoded? What do columns represent, 
makes sure they are correctly labelled. How is the data indexed. 
Crete visualisation routines to assess the data (e.g. in bokeh). 
Ensure that date formats are correct and correctly timezoned.
"""


def get_tags():
    # Retrieve POIs
    tags = {"amenity": ["pub", "kindergarten", "library", "school", "university",
                        "bus_station", "parking", "taxi", "bank", "hospital", "cinema"],
            "leisure": ["adult_gaming_centre", "amusement_arcade", "bench_resort",
                        "dog_park", "fitness_centre", "garden", "park", "playground",
                        "sports_centre"],
            "shop": ["alcohol", "bakery", "dairy", "ice_cream", "tea", "water"]
            }

    return tags


def get_prop_types():
    prop_types = ['F', 'D', 'S', 'O', 'T']
    return prop_types


def get_place_name(house_loc):
    # assuming we are all in the same district
    if len(house_loc['district'].unique()) > 1:
        print("bounding box large, including multiple districts")
    return house_loc['district'].unique()[0] + ", United Kingdom"



def draw_location(conn, latitude, longitude, date, place_name,
                  box_width=0.02, box_height=0.02):

    place_stub = place_name.lower().replace(' ', '-').replace(',', '')

    north = latitude + box_height
    south = latitude - box_height
    west = longitude - box_width
    east = longitude + box_width

    graph = ox.graph_from_bbox(north, south, east, west)

    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)

    # Get place boundary related to the place name as a geodataframe
    area = ox.geocode_to_gdf(place_stub)

    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [50, 1]})
    fig.set_size_inches(13.5, 10.5)

    # Plot the footprint
    area.plot(ax=ax[0], facecolor="white")

    # Plot street edges
    edges.plot(ax=ax[0], linewidth=1, edgecolor="lightgray")

    ax[0].set_xlim([west, east])
    ax[0].set_ylim([south, north])
    ax[0].set_xlabel("longitude")
    ax[0].set_ylabel("latitude")

    pois = access.get_osm_pois(latitude, longitude, place_name, box_width, box_height)

    house_loc = access.price_data_with_date_location(conn, latitude, longitude, date,
                                              box_height, box_width, date_range=180)
    print(f"there are {len(house_loc)} number of houses")

    # Plot all POIs
    pois.plot(ax=ax[0], color="lightskyblue", alpha=0.5, markersize=10)

    # plot all houses

    # low price -> yellow, mid -> green, high -> red

    proptype_mark = {
        "F": ".",
        "S": "d",
        "D": "*",
        "T": "v",
        "O": "s"
    }


    norm = plt.Normalize(min(house_loc['price']), max(house_loc['price']))

    for t in house_loc['property_type'].unique():
        house_loc_temp = house_loc[house_loc['property_type'] == t]

        ax[0].scatter(
            x=house_loc_temp['longitude'], y=house_loc_temp['latitude'],
            c=house_loc_temp['price'].values,
            norm=norm,
            marker=proptype_mark[t], s=50, alpha=0.7, zorder=5)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'),
                 cax=ax[1],
                 orientation='vertical',
                 label='prices')

    plt.tight_layout()
    plt.show()



def retrieve_houses(conn, latitude, longitude, date, property_type):
    # Retrieve POIs

    init_box_height = 0.01
    init_box_width = 0.01
    init_date_range = 30

    box_height = init_box_height
    box_width = init_box_width
    date_range = init_date_range

    # get the prices_coordinate_data
    house_loc = None
    for i in range(1, 10):
        house_loc = access.price_data_with_date_location(conn, latitude, longitude, date,
                                                         box_height, box_width, date_range)

        retrieved_properties = house_loc['property_type'].unique()
        if house_loc.shape[0] > 50 and property_type in retrieved_properties:
            break
        else:
            box_height += 0.01 * i
            box_width += 0.01 * i
            date_range += 5 * i
            if (house_loc.shape[0] < 50):
                print(f"found only {house_loc.shape[0]} number of houses")
            elif not property_type in retrieved_properties:
                print(f"have not found property {property_type} from dataset")

            print(f"increasing bounding box size and date range to \
            width=height={box_height} date range={date_range}")

    if house_loc.shape[0] < 50:
        print(f"WARNING too few houses {house_loc.shape[0]} around to train")
    if not property_type in retrieved_properties:
        print(f"WARNING did not find property {property_type} from dataset\
                prediction unlikely to be accurate")

    print(f"number of houses found {house_loc.shape[0]} and number of {property_type}\
            houses in the dataset is \
            {house_loc[house_loc['property_type'] == property_type].shape[0]}")

    return house_loc



def retrieve_pois(latitude, longitude, house_loc):
    # get OSM data

    init_box_height = 0.02
    init_box_width = 0.02

    box_height = init_box_height
    box_width = init_box_width

    place_name = get_place_name(house_loc)
    print(f"the place we are in is {place_name}")

    pois = None
    for i in range(1, 5):
        pois = access.get_osm_pois(latitude, longitude, box_width, box_height)
        if pois.shape[0] > 50:
            break
        else:
            box_height += 0.01 * i
            box_width += 0.01 * i
            print(f"found only {pois.shape[0]} number of pois")
            print(
                f"increasing bounding box size to width=height={box_height} to get more pois")

    print(f"finally found {pois.shape[0]} number of pois")

    print(
        f"Actual types of amenities being retrieved {pois['amenity'].unique()}")

    print("There are {number} points of interest surrounding {placename} latitude: \
            {latitude}, longitude: {longitude}".format(number=len(pois),
                                                       placename=place_name, latitude=latitude, longitude=longitude))

    return pois


def data():
    """Load the data from access and ensure missing values are correctly encoded 
    as well as indices correct, column names informative, date and times correctly 
    formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    # the data is quite clean and don't think need to cleaning at this stage
    return df


def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data):
    """Provide a view of the data that allows the user to verify 
    some aspect of its quality."""
    place_name = "SOUTH LAKELAND, United Kingdom"
    latitude = 54.4 
    longitude = -2.9 

    date = '2018-04-26'
    draw_location(access.get_conn(), latitude, longitude, date, place_name)


def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    # the data returned by data() is itself a labelled data and will be processed
    # later on by the train function
    return data()
