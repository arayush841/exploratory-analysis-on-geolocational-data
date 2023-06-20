import numpy as np
import seaborn as sns
import pandas as pd
from geopy.geocoders import Nominatim
import requests
from pandas.io.json import json_normalize
import minisom
import scipy
from sklearn import preprocessing, cluster
import geopy
import folium
import matplotlib.pyplot as plt


data = pd.read_csv("Dataset/food_coded.csv")


# Storing only the necessery columns in column
column = ['cook', 'eating_out', 'employment', 'ethnic_food', 'exercise',
          'fruit_day', 'income', 'on_off_campus', 'pay_meal_out', 'sports', 'veggies_day']

d = data[column]

# Drop NA values
s = d.dropna()

# Run KMeans Clustering on the data

f = ['cook', 'income']
X = s[f]
max_k = 10
# iterations
distortions = []
for i in range(1, max_k+1):
    if len(X) >= i:
        model = cluster.KMeans(
            n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        model.fit(X)
        distortions.append(model.inertia_)
# best k: the lowest derivative
k = [i*100 for i in np.diff(distortions, 2)].index(min([i*100 for i
                                                        in np.diff(distortions, 2)]))


CLIENT_ID = "KTCJJ2YZ2143QHEZ2JAQS4FJIO5DLSDO0YN4YBXPMI5NKTEF"  # your Foursquare ID
# your Foursquare Secret
CLIENT_SECRET = "KNG2LO22BPLHN1E3OAHWLYQ5PQBN14XYZMEMAS0CPJEJKOTR"
VERSION = '20200316'
LIMIT = 10000


city = "Delhi"
# get location
locator = geopy.geocoders.Nominatim(user_agent="MyCoder")
location = locator.geocode(city)
print(location)
# keep latitude and longitude only
location = [location.latitude, location.longitude]
print("[lat, long]:", location)


url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID,
    CLIENT_SECRET,
    VERSION,
    # 17.448372, 78.526957, #Hyderabad
    # 12.9716, 77.5946, #Bangalore
    # 28.4595, 77.0266, #Gurugram
    # 28.7041, 77.1025,  # Delhi
    # 18.5204 ,73.8567, #Pune
    location[0], location[1],
    30000,
    LIMIT)

results = requests.get(url).json()


venues = results['response']['groups'][0]['items']
nearby_venues = pd.json_normalize(venues)


"""## Adding two more Columns Restaurant and Others
 

1.   Restaurant: Number of Restaurant in the radius of 20 km
2.   others:Number of Gyms, Parks,etc in the radius of 20 km



"""

resta = []
oth = []
for lat, long in zip(nearby_venues['venue.location.lat'], nearby_venues['venue.location.lng']):
    url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
        CLIENT_ID,
        CLIENT_SECRET,
        VERSION,
        lat, long,
        1000,
        100)
    res = requests.get(url).json()
    venue = res['response']['groups'][0]['items']
    nearby_venue = pd.json_normalize(venue)
    df = nearby_venue['venue.categories']

    g = []
    for i in range(0, df.size):
        g.append(df[i][0]['icon']['prefix'].find('food'))
    co = 0
    for i in g:
        if i > 1:
            co += 1
    resta.append(co)
    oth.append(len(g)-co)

nearby_venues['restaurant'] = resta
nearby_venues['others'] = oth
nearby_venues


"""## Changing the Column Name"""

lat = nearby_venues['venue.location.lat']
long = nearby_venues['venue.location.lng']

"""## Install the minisom library using pip

MiniSom is a minimalistic and Numpy based implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Network able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display. Minisom is designed to allow researchers to easily build on top of it and to give students the ability to quickly grasp its details.
"""


"""## Run K Means clustering on the dataset, with the optimal K value using Elbow Method

A fundamental step for any unsupervised algorithm is to determine the optimal number of clusters into which the data may be clustered. The Elbow Method is one of the most popular methods to determine this optimal value of k.
"""

f = ['venue.location.lat', 'venue.location.lng']
X = nearby_venues[f]
max_k = 10
# iterations
distortions = []
for i in range(1, max_k+1):
    if len(X) >= i:
        model = cluster.KMeans(
            n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        model.fit(X)
        distortions.append(model.inertia_)
# best k: the lowest derivative
k = [i*100 for i in np.diff(distortions, 2)].index(min([i*100 for i
                                                        in np.diff(distortions, 2)]))


n = nearby_venues.drop(['referralId', 'reasons.count', 'reasons.items', 'venue.id',
                        'venue.name',
                        'venue.location.labeledLatLngs', 'venue.location.distance',
                        'venue.location.cc',
                        'venue.categories', 'venue.photos.count', 'venue.photos.groups',
                        'venue.location.crossStreet', 'venue.location.address', 'venue.location.city',
                        'venue.location.state', 'venue.location.crossStreet',
                        'venue.location.neighborhood', 'venue.venuePage.id',
                        'venue.location.postalCode', 'venue.location.country'], axis=1)

"""## Dropping Nan Values from Dataset"""

n = n.dropna()
n = n.rename(columns={'venue.location.lat': 'lat',
             'venue.location.lng': 'long'})

"""###Convert Every Row of Column ***'venue.location.formattedAddress'*** from List to String"""

n['venue.location.formattedAddress']

spec_chars = ["[", "]"]
for char in spec_chars:
    n['venue.location.formattedAddress'] = n['venue.location.formattedAddress'].astype(
        str).str.replace(char, ' ')

n

"""#Plot the clustered locations on a map"""

x, y = "lat", "long"
color = "restaurant"
size = "others"
popup = "venue.location.formattedAddress"
data = n.copy()

# create color column
lst_colors = ["red", "green", "orange"]
lst_elements = sorted(list(n[color].unique()))

# create size column (scaled)
scaler = preprocessing.MinMaxScaler(feature_range=(3, 15))
data["size"] = scaler.fit_transform(
    data[size].values.reshape(-1, 1)).reshape(-1)

# initialize the map with the starting location
map_ = folium.Map(location=location, tiles="cartodbpositron",
                  zoom_start=11)
# add points
data.apply(lambda row: folium.CircleMarker(
           location=[row[x], row[y]], popup=row[popup],
           radius=row["size"]).add_to(map_), axis=1)
# add html legend


X = n[["lat", "long"]]
max_k = 10
# iterations
distortions = []
for i in range(1, max_k+1):
    if len(X) >= i:
        model = cluster.KMeans(
            n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        model.fit(X)
        distortions.append(model.inertia_)
# best k: the lowest derivative
k = [i*100 for i in np.diff(distortions, 2)
     ].index(min([i*100 for i in np.diff(distortions, 2)]))

k = 5
model = cluster.KMeans(n_clusters=k, init='k-means++')
X = n[["lat", "long"]]
# clustering
dtf_X = X.copy()
dtf_X["cluster"] = model.fit_predict(X)
# find real centroids
closest, distances = scipy.cluster.vq.vq(model.cluster_centers_,
                                         dtf_X.drop("cluster", axis=1).values)
dtf_X["centroids"] = 0
for i in closest:
    dtf_X["centroids"].iloc[i] = 1
# add clustering info to the original dataset
n[["cluster", "centroids"]] = dtf_X[["cluster", "centroids"]]


# plot
fig, ax = plt.subplots()
sns.scatterplot(x="lat", y="long", data=n,
                palette=sns.color_palette("bright", k),
                hue='cluster', size="centroids", size_order=[1, 0],
                legend="brief", ax=ax).set_title('Clustering (k='+str(k)+')')
th_centroids = model.cluster_centers_
ax.scatter(th_centroids[:, 0], th_centroids[:, 1], s=50, c='black',
           marker="x")

model = cluster.AffinityPropagation()

k = n["cluster"].nunique()
sns.scatterplot(x="lat", y="long", data=n,
                palette=sns.color_palette("bright", k),
                hue='cluster', size="centroids", size_order=[1, 0],
                legend="brief").set_title('Clustering (k='+str(k)+')')

x, y = "lat", "long"
color = "cluster"
size = "restaurant"
popup = "venue.location.formattedAddress"
marker = "centroids"
data = n.copy()
# create color column
lst_elements = sorted(list(n[color].unique()))
lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in
              range(len(lst_elements))]
data["color"] = data[color].apply(lambda x:
                                  lst_colors[lst_elements.index(x)])
# create size column (scaled)
scaler = preprocessing.MinMaxScaler(feature_range=(3, 15))
data["size"] = scaler.fit_transform(
    data[size].values.reshape(-1, 1)).reshape(-1)
# initialize the map with the starting location
map_ = folium.Map(location=location, tiles="cartodbpositron",
                  zoom_start=11)
# add points
data.apply(lambda row: folium.CircleMarker(
           location=[row[x], row[y]],
           color=row["color"], fill=True, popup=row[popup],
           radius=row["size"]).add_to(map_), axis=1)
# add html legend
legend_html = """<div style="position:fixed; bottom:10px; left:10px; border:2px solid black; z-index:9999; font-size:14px;">&nbsp;<b>"""+color+""":</b><br>"""
for i in lst_elements:
    legend_html = legend_html+"""&nbsp;<i class="fa fa-circle 
     fa-1x" style="color:"""+lst_colors[lst_elements.index(i)]+"""">
     </i>&nbsp;"""+str(i)+"""<br>"""
legend_html = legend_html+"""</div>"""
map_.get_root().html.add_child(folium.Element(legend_html))
# add centroids marker
lst_elements = sorted(list(n[marker].unique()))
data[data[marker] == 1].apply(lambda row:
                              folium.Marker(location=[row[x], row[y]],
                                            draggable=False,  popup=row[popup],
                                            icon=folium.Icon(color="black")).add_to(map_), axis=1)

# map_.show_in_browser()
map_.save("Cities/Chandigarh.html")
