# Exploratory Analysis on Geolocational Data

This project helps a user find restaurants in a City. The bigger the size of circle more the number of restaurants present at that location.

This project uses K-means clustering to cluster a group of amenities and find their centroid. 

Steps:
1. Data Collection: Collected data from Kaggle.
2. Data Cleaning: Removal of Null and unnecessary values.
3. Data Visualisation: Visualised cleaned data for a better understanding of the dataset.
4. Run K-Means clustering: Employed elbow method to find the optimal value of K which represents the number of clusters present in the dataset.
5. Fetching Geolocational Data: Firstly, initialized GeoPy, a library to fetch the coordinates of a city, later the coordinates were passed to FourSquare API which lists the venues present in that particular city. The data received is in JSON format, which is converted into a data frame for further processing.
6. Presenting findings on a Map: Using the Folium library, the acquired data were plotted on a map. Each represented an area(with address) where the number of restaurants was more.

Libraries used:
1. Numpy
2. Pandas
3. scikit-learn
4. Tkinter
5. GeoPy
6. Folium
7. Seaborn
8. Webview

How to run this project?
1. Open this folder in Visual Studio Code.
2. Open the project.py file.
3. Ensure you have VS Code extension Live Server(by Ritwick Dey) installed.
4. Click on Go Live present on the bottom right of your VSC window. This opens an instance of your browser, we don't need that so close!
5. Run the project.py file.
6. Select any city from the list of cities available.
7. Click on, Go.
8. This opens the map of the city you've selected and restaurants are plotted on that map.

What if want to add a new city?
1. Go to the exploratory_data_analysis_on_geolocational_data.py file.
2. On line 52 you'll see a variable named city, enter the city name.
3. Now go to line 296, there on the map_.save("Cities/Chandigarh.html"), and change the city name according to your needs.
4. Run the .py file.
5. The city's map will get stored in the Cities folder.
6. Now open the project.py file.
7. Add your city name in the cities tuple(recommended to name the city the same as the saved map file name).
8. Run the project.py file.

