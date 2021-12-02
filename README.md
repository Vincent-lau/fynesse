# Fynesse Template

Below I highlight some aspects that I think is worth mentioning for this project.
The overall structure follows the fynesse template as described, separating
access, assess and address of the problem. The high level idea is that access
sets up the database and uploading and downloading of various data. Assess looks
at the map and house data and tries to plot them. Finally address fits a model
on price against features.

## Access

There are three major datasets used in this project, here are some attributions.

The price paid data
Contains HM Land Registry data © Crown copyright and database right 2021.
This data is licensed under the Open Government Licence v3.0.

And the address data contained in the price paid data is licensed as
Royal Mail and Ordnance Survey permit your use of Address Data in the Price Paid Data:
for personal and/or non-commercial use

The Postcode data is
Free to use for any purpose - attribution required.
Contains OS data © Crown copyright and database right
Contains Royal Mail data © Royal Mail copyright and database right
Source: Office for National Statistics licensed under the Open Government Licence v.3.0

And the OSM data
© OpenStreetMap contributors
[License page](https://www.openstreetmap.org/copyright)

As far as I am aware, all of these uses are legal according to the licenses respectively.
And I don't see much ethical issues either from using this dataset.

Now onto the codebase. The main functions in `access.py` are database setup code
and data download/upload using libraries such as `pymysql` etc. There is the
`price_data_with_date_location` function which does the join of price data and
postcode data.

## Assess

Taking a quick look at the price paid data, postcode data and the OSM data, they
are fairly clean to start with. The pois has some `NaN` as place names but some
of them are supposed to encode as `NaN` to indicate that they are not a amenity
and leisure at the same time.

Some reusable code that can be used independent of the problem to be addressed
is the `draw_location()` function, which draws houses on the map with colour
mappin for prices along with other pois.

## Address

To build a house price predicting model, I examined what features are the most
suitable by looking at distances to various amenities and selected those with
the highest correlation coefficient. And also used the property_type encoded
using one-hot encoding. I fitted a Gaussian regression model using `OLS` from
`statsmodels` on the data and
did the validation by train/test data split and tested using MSE. In `address.py`
there is the `predict_price()` function which is the top level prediction, and also
`feature_selection()` which looks at distances to different places and find the
most correalted ones.
