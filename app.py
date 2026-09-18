from flask import Flask, render_template, request
from model import predict_fare
from datetime import datetime
import math

app = Flask(__name__)

location_coords = {
    "T Nagar": (13.0418, 80.2341),
    "Velachery": (12.9750, 80.2210),
    "Anna Nagar": (13.0850, 80.2101),
    "Tambaram": (12.9249, 80.1275),
    "Adyar": (13.0012, 80.2565),
    "Guindy": (13.0067, 80.2206),
    "Porur": (13.0356, 80.1586),
    "OMR": (12.9170, 80.2300),
    "Perungudi": (12.9716, 80.2470),
    "Sholinganallur": (12.9010, 80.2279),
    "Chromepet": (12.9516, 80.1462),
    "Egmore": (13.0732, 80.2609),
    "Central": (13.0827, 80.2707),
    "Besant Nagar": (13.0003, 80.2667),
    "Thiruvanmiyur": (12.9856, 80.2592),
    "Medavakkam": (12.9229, 80.1920),
    "Tambaram East": (12.9240, 80.1300),
    "Tambaram West": (12.9200, 80.1200),
    "Ambattur": (13.1143, 80.1548),
    "Avadi": (13.1147, 80.1098)
}

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)

    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dLon/2)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pickup = request.form["pickup_location"]
    drop = request.form["drop_location"]
    passengers = float(request.form["passenger_count"])
    model_type = request.form["model"]

    if pickup == drop:
        return render_template("index.html", prediction="Same location ❌")

    lat1, lon1 = location_coords[pickup]
    lat2, lon2 = location_coords[drop]

    distance = calculate_distance(lat1, lon1, lat2, lon2)

    now = datetime.now()

    features = [
        lat1, lon1, lat2, lon2,
        passengers, distance,
        now.hour, now.month, now.weekday()
    ]

    result = predict_fare(features, model_type)

    # FINAL FARE
    fare = (result[0] * 80) + 50
    if fare < 100:
        fare = 100

    base = 50
    remaining = fare - base

    distance_cost = remaining * 0.7
    passenger_cost = remaining * 0.3

    return render_template(
        "index.html",
        prediction=f"{model_type.upper()} → Estimated Fare: ₹ {round(fare,2)}",
        base=round(base,2),
        distance_cost=round(distance_cost,2),
        passenger_cost=round(passenger_cost,2),
        total=round(fare,2)
    )

if __name__ == "__main__":
    app.run(debug=True)


