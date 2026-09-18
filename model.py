import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

# 📊 LOAD DATA
data = pd.read_csv("nyc_taxi_dataset_100k_with_nulls.csv")

# 🧹 HANDLE NULLS 
data.fillna({
    "pickup_latitude": 40.681089,
    "pickup_longitude": -73.9690,
    "dropoff_latitude": 40.7520,
    "dropoff_longitude": -74.107021,
    "passenger_count": 3.0,
    "trip_distance_miles": 14.66
}, inplace=True)

# ⏰ DATETIME FEATURES
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data['hour'] = data['pickup_datetime'].dt.hour
data['month'] = data['pickup_datetime'].dt.month
data['weekday'] = data['pickup_datetime'].dt.weekday

data.drop('pickup_datetime', axis=1, inplace=True)

# 🎯 FEATURES & TARGET
X = data.drop("fare_amount", axis=1)
y = data["fare_amount"]

# 🧠 TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 TRAIN MULTIPLE MODELS
linear = LinearRegression()
linear.fit(X_train, y_train)

tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)

forest = RandomForestRegressor()
forest.fit(X_train, y_train)

poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly.fit(X_train, y_train)

# 🔮 PREDICTION FUNCTION
def predict_fare(features, model_type):
    features = np.array(features).reshape(1, -1)

    if model_type == "linear":
        result = linear.predict(features)
    elif model_type == "tree":
        result = tree.predict(features)
    elif model_type == "forest":
        result = forest.predict(features)
    elif model_type == "poly":
        result = poly.predict(features)
    else:
        result = linear.predict(features)  # default

    return result