from matplotlib.pyplot import connect
from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import mongodb
import sqlite"""

import pymysql
import urllib.request
import datetime
import pandas as pd
import zipfile
import osmnx as ox
from . import assess

# This file accesses the data

"""
Place commands in this file to access the data electronically. Don't remove
any missing values, or deal with outliers. Make sure you have legalities correct,
both intellectual property and personal data privacy rights. Beyond the legal side
also think about the ethical issues around this data.

"""


# Insert your database url below
db_details = {"url": "database-sl955.cgrre17yxw11.eu-west-2.rds.amazonaws.com",
                    "port": 3306}


# RUN ONCE to download dataset

def download_postcode_data():
    urllib.request.urlretrieve(f'https://www.getthedata.com/downloads/open_postcode_geo.csv.zip',
                               f'drive/MyDrive/dataset/open_postcode_geo.csv.zip')
    with zipfile.ZipFile('drive/MyDrive/dataset/open_postcode_geo.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('drive/MyDrive/dataset/open_postcode_geo.csv')


def download_pp_data():
    for year in range(1995, 2022):
        for part in range(1, 3):
            urllib.request.urlretrieve(
                f'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1\
                .amazonaws.com/pp-{str(year)}-part{str(part)}.csv', f'drive/My\
                Drive/dataset/pp-{str(year)}-part{str(part)}.csv')
            print(f'Downloaded {str(year)} part {str(part)}')


def create_pp_db(conn):
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS `pp_data`;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS `pp_data` (
            `transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,
            `price` int(10) NOT NULL,
            `date_of_transfer` date NOT NULL,
            `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
            `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
            `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
            `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
            `primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
            `secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
            `street` tinytext COLLATE utf8_bin NOT NULL,
            `locality` tinytext COLLATE utf8_bin NOT NULL,
            `town_city` tinytext COLLATE utf8_bin NOT NULL,
            `district` tinytext COLLATE utf8_bin NOT NULL,
            `county` tinytext COLLATE utf8_bin NOT NULL,
            `ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,
            `record_status` varchar(2) COLLATE utf8_bin NOT NULL,
            `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT PRIMARY KEY
            ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;
        """.replace("\n", " "))
        cur.execute("""
         CREATE INDEX `pp.postcode` USING HASH
            ON `pp_data`
                (postcode);
        """.replace("\n", " "))
        cur.execute("""
            CREATE INDEX `pp.date` USING HASH
            ON `pp_data`
                (date_of_transfer)
        """.replace("\n", " ").replace("\n", " "))
    conn.commit()


def upload_pp_data(conn, start_year, end_year, filepath = 'drive/MyDrive/dataset/'):

    for year in range(start_year, end_year):
        for part in range(1, 3):
            filename = filepath + f'pp-{str(year)}-part{str(part)}.csv'
            with conn.cursor() as cur:
                sql = f"""LOAD DATA LOCAL INFILE '{filename}' INTO TABLE pp_data
                            FIELDS
                            TERMINATED BY ','
                            ENCLOSED BY '"'
                            LINES STARTING BY '' TERMINATED BY
                        """
                ending = "'\n';"
                cur.execute(sql.replace("\n", " ") + ending)
                print(f"{filename} loaded into databases")
    conn.commit()
    print("dateset committed")


def create_postcode_data(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS `postcode_data` (
            `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
            `status` enum('live','terminated') NOT NULL,
            `usertype` enum('small', 'large') NOT NULL,
            `easting` int unsigned,
            `northing` int unsigned,
            `positional_quality_indicator` int NOT NULL,
            `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
            `latitude` decimal(11,8) NOT NULL,
            `longitude` decimal(10,8) NOT NULL,
            `postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,
            `postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,
            `postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,
            `postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,
            `postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,
            `postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,
            `outcode` varchar(4) COLLATE utf8_bin NOT NULL,
            `incode` varchar(3)  COLLATE utf8_bin NOT NULL,
            `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT PRIMARY KEY
            ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin; 
        """.replace("\n", " ")
                    )
        cur.execute("""
            CREATE INDEX `po.postcode` USING HASH
                ON `postcode_data`
                (postcode); 
        """.replace("\n", " "))

    conn.commit()
    print("postcode table created")


def upload_postcode_data(conn):
    filepath = 'drive/MyDrive/dataset/'
    with conn.cursor() as cur:
        cur.execute(f"""
            LOAD DATA LOCAL INFILE '{filepath}open_postcode_geo/open_postcode_geo.csv' INTO TABLE `postcode_data`
            FIELDS TERMINATED BY ',' 
            LINES STARTING BY '' TERMINATED BY '\n'; 
        """
                    )
    conn.commit()
    print("postcode data loaded")


def create_prices_coordinate_data(conn):
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS `prices_coordinates_data`;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS `prices_coordinates_data` (
            `price` int(10) unsigned NOT NULL,
            `date_of_transfer` date NOT NULL,
            `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
            `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
            `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
            `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
            `locality` tinytext COLLATE utf8_bin NOT NULL,
            `town_city` tinytext COLLATE utf8_bin NOT NULL,
            `district` tinytext COLLATE utf8_bin NOT NULL,
            `county` tinytext COLLATE utf8_bin NOT NULL,
            `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
            `latitude` decimal(11,8) NOT NULL,
            `longitude` decimal(10,8) NOT NULL,
            `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT PRIMARY KEY
            ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;
        """.replace("\n", " "))
    conn.commit()
    print("prices coordinate data created")


def connect_db(cred_path = "credentials.yaml", database_details = db_details):
    with open(cred_path) as file:
        credentials = yaml.safe_load(file)
    return create_connection(user=credentials["username"],
                             password=credentials["password"],
                             host=database_details["url"],
                             database="property_prices",
                             port = database_details["port"])


def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")

    # initialise the db
    return initialise_db(conn)


def initialise_db(conn):
    with conn.cursor() as cur:
        cur.execute('SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";')
        cur.execute('SET time_zone = "+00:00";')

        cur.execute('CREATE DATABASE IF NOT EXISTS `property_prices` DEFAULT \
        CHARACTER SET utf8 COLLATE utf8_bin;')
        cur.execute('USE `property_prices`;')

    conn.commit()
    return conn


def price_data_with_date_location(conn, latitude, longitude, date, box_height=0.2,
                                  box_width=0.2, date_range=90):

    d1 = datetime.datetime.strptime(date, "%Y-%m-%d")
    d2 = d1 + datetime.timedelta(days=date_range)

    d1 = date
    d2 = d2.strftime("%Y-%m-%d")

    df = pd.DataFrame()
    with conn.cursor() as cur:
        cur.execute(f"""
      select price, date_of_transfer, pp_data.postcode as postcode, property_type,
        new_build_flag, tenure_type, locality, town_city, district, county, country,
        latitude, longitude
      from pp_data 
      inner join postcode_data
      on pp_data.postcode = postcode_data.postcode
      where
        latitude between {latitude} - {box_height} and  {latitude} + {box_height} and
        longitude between {longitude} - {box_width} and {longitude} + {box_width} and
        date_of_transfer between '{d1}' and '{d2}'
    """.replace("\n", " "))

        rows = cur.fetchall()

        df = pd.DataFrame(rows, columns=["price", "date_of_transfer", "postcode", "property_type",
                                         "new_build_flag", "tenure_type", "locality", "town_city", "district", "county", "country",
                                         "latitude", "longitude"])

    return df


def select_top(conn, table,  n):
    """
    Query n first rows of the table
    :param conn: the Connection object
    :param table: The table to query
    :param n: Number of rows to query
    """
    with conn.cursor() as cur:
        cur.execute(f'SELECT * FROM {table} LIMIT {n}')

    rows = cur.fetchall()
    return rows


def head(conn, table, n=6):
    rows = select_top(conn, table, n)
    for r in rows:
        print(r)


def price_data_with_date_location(conn, latitude, longitude, date, box_height=0.2, 
                                  box_width=0.2, date_range=90):

  d1 = datetime.datetime.strptime(date, "%Y-%m-%d")
  d2 = d1 + datetime.timedelta(days = date_range)
  
  d1 = date
  d2 = d2.strftime("%Y-%m-%d")     

  df = pd.DataFrame()
  with conn.cursor() as cur:
    cur.execute(f"""
      select price, date_of_transfer, pp_data.postcode as postcode, property_type,
        new_build_flag, tenure_type, locality, town_city, district, county, country,
        latitude, longitude
      from pp_data 
      inner join postcode_data
      on pp_data.postcode = postcode_data.postcode
      where
        latitude between {latitude} - {box_height}  and  {latitude} + {box_height} and
        longitude between {longitude} - {box_width}  and {longitude} + {box_width} and
        date_of_transfer between '{d1}' and '{d2}'
    """.replace("\n", " "))

    rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["price", "date_of_transfer", "postcode" , "property_type",
      "new_build_flag", "tenure_type", "locality", "town_city", "district", "county", "country",
      "latitude", "longitude"])

    # df.set_index('db_id', inplace=True)

  return df

def get_osm_pois(latitude, longitude, box_width=0.02, box_height=0.02):

  north = latitude + box_height
  south = latitude - box_height
  west = longitude - box_width
  east = longitude + box_width

  pois = ox.geometries_from_bbox(north, south, east, west, assess.get_tags())

  return pois

conn = None

def get_conn():
    if conn == None:
        return connect_db()
    else:
        return conn

def data():
    """Read the data from the web or local file, 
    returning structured format such as a data frame"""
    raise NotImplementedError
