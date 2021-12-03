# UK House price prediction with fynesse

The overall structure of this project follows the fynesse template as required,
separating access, assess and address of the problem. The high level idea is that access
sets up the database and uploading and downloading of various data. Assess looks
at each dataset in term (`pp_data`, `postcode_data` and osm data) to see their
representation is reasonable and can be used for model fitting. And then plotting
the map and house data together to see if there is any correlation.
Finally, address fits a Gaussian model on price against features and did some simple
validation of its effectiveness.

## Access

There are three major datasets that are used in this project, here are some attributions.

- Price paid data

Contains HM Land Registry data © Crown copyright and database right 2021.
This data is licensed under the Open Government Licence v3.0.

And the address data contained in the price paid data is licensed as
Royal Mail and Ordnance Survey permit your use of Address Data in the Price Paid Data:
for personal and/or non-commercial use

- Postcode data

Free to use for any purpose - attribution required.
Contains OS data © Crown copyright and database right
Contains Royal Mail data © Royal Mail copyright and database right
Source: Office for National Statistics licensed under the Open Government Licence v.3.0

- OSM data

© OpenStreetMap contributors
[License page](https://www.openstreetmap.org/copyright)

As far as I am aware, all of these uses are legal according to the licenses respectively.
And I don't see any ethical issues either from using this dataset.

Now onto the codebase. The main functions in `access.py` are database setup code
and data download/upload using libraries such as `pymysql` etc. There is the
`price_data_with_date_location` function which does the join of price data and
postcode data.

## Assess

Here I took a quick look at the price paid data, postcode data and the OSM data, they
are fairly clean to start with, although the postcode data does have a `NaN` in
a place that can cause trouble for algorithms later on.
The pois has some `NaN` as place names but some of them are supposed to encode as
`NaN` to indicate, for example, that a node cannot be a amenity and leisure at the same time.

Some reusable code that can be used independent of the problem to be addressed
is the `draw_location()` function, which draws houses on the map with colour
mappin for prices along with other pois. And the `price_data_with_date_location()`
which does the minimum data cleaning job of getting rid of the `NaN` value. The
`data_postcode_cleaned` is not used in this project but can be useful as well
as it cleans the postcode dataset.

## Address

To build a house price predicting model, I examined what features are the most
suitable by looking at distances to various amenities and selected those with
the highest correlation coefficient. I also used the property_type as one of the
features after looking at its influence on house prices.
I fitted a Gaussian regression model using `OLS` from `statsmodels` on the data and
did the validation by train/test data split and measured the MSE of the model on
the test set. In `address.py`
there is the `predict_price()` function which is the top level prediction, and also
`feature_selection()` which looks at distances to different places and find the
most correalted ones. There is also `train()` and `validate()` whose name is
pretty self-explanatory.

I am sure there are a ton of other useful features that I have never thought about,
or even some that sound promising but did not have time to look into, such as
distances to city centre, time from a particular point, etc.

## Thinking about tests

During the time I refactored my code, I find myself spending quite a bit of time
clicking through each cell in the notebook, testing whether the refactor break
any existing functionalties. There is merit in automating this process, I believe.
Stealing ideas from software engineering, regression test can be useful by simply
asking each function to do a simple task and then each time the codebase changes
we can see if our functions are still functioning.

Sanity checks can be useful as well, I encountered some negative value prediction
while I was writing my prediction function and did not notice that until later on
because I was overwhelmed by the bunch of other outputs. So sanity check can be
useful for some functions.

"Progression" test may not be useful for the address part as this part is all
about trying to address a particular question, but might be useful for access
and assess. For access, we might want to check availability of the data, although
this can be hard to automate if there is legal/ethical issues. For assess, we
might want to do some fuzz testing to protect against feature invalid values
that are going to be included in the dataset.
