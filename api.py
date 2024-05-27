# reviews
import calendar
from datetime import date, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import holidays
from reviews import model, Reviewdata
import sales
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import json

# Define API endpoints

app = Flask(__name__)
CORS(app)

@app.route('/api/reviews', methods=['POST'])
def analyze_reviews():
    data = request.get_json()
    if not data or 'reviews' not in data:
        return jsonify({'error': 'No data found or incorrect format'})
    reviews = data['reviews']
    results = model.predict(reviews)
    return jsonify({'results': results.tolist()})

@app.route('/api/forecast', methods=['GET'])
def predict_consumption():
    json_file_path = "/Users/aahanaganjewar/Desktop/rohit/reviewsAPI/hello.json"

    # Load data from the JSON file with the 'utf-8' encoding
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Convert the data
    item_dfs = {}
    for date, items in data.items():
        for item, consumption in items.items():
            if item not in item_dfs:
                item_dfs[item] = {'date': [], 'consumption': []}
            item_dfs[item]['date'].append(date)
            item_dfs[item]['consumption'].append(consumption)


    forecasts = {}
    for item, item_data in item_dfs.items():
        df = pd.DataFrame(item_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df_resampled = df.resample('D').sum()

    
        model = ARIMA(df_resampled['consumption'], order=(5, 1, 2)) # You may need to adjust order
        fit_model = model.fit()

        # Forecast future values
        forecast = fit_model.get_forecast(steps=7) 
        forecasts[item] = forecast.predicted_mean[-1] 



    # Assuming forecast_df is already created as mentioned in the question
    forecast_df = pd.DataFrame(list(forecasts.items()), columns=['Item', 'Predicted Consumption'])
    print("--------------------------------------------")
    print(forecast_df)
    print("--------------------------------------------")
    # Convert DataFrame to JSON
    json_forecast = forecast_df.to_json(orient='records')

    # Convert dictionary to JSON using jsonify
    return json_forecast


@app.route('/api/happy_reviews', methods=['GET'])
def get_happy_reviews():
    happy_reviews = Reviewdata[Reviewdata['Is_Response']
                               == 'happy']['cleaned_description_new'].tolist()
    return jsonify({'happy_reviews': happy_reviews})


@app.route('/api/not_happy_reviews', methods=['GET'])
def get_not_happy_reviews():
    not_happy_reviews = Reviewdata[Reviewdata['Is_Response']
                                   == 'not happy']['cleaned_description_new'].tolist()
    return jsonify({'not_happy_reviews': not_happy_reviews})


@app.route('/api/percentage_happy_reviews', methods=['GET'])
def get_percentage_happy_reviews():
    total_reviews = len(Reviewdata)
    happy_reviews_count = len(Reviewdata[Reviewdata['Is_Response'] == 'happy'])
    percentage_happy_reviews = (happy_reviews_count / total_reviews) * 100
    return jsonify({'percentage_happy_reviews': percentage_happy_reviews})


@app.route('/api/percentage_not_happy_reviews', methods=['GET'])
def get_percentage_not_happy_reviews():
    total_reviews = len(Reviewdata)
    not_happy_reviews_count = len(
        Reviewdata[Reviewdata['Is_Response'] == 'not happy'])
    percentage_not_happy_reviews = (
        not_happy_reviews_count / total_reviews) * 100
    return jsonify({'percentage_not_happy_reviews': percentage_not_happy_reviews})


# sales


# Define global variables for today, calendar, and us_holidays
today = date.today()
us_holidays = holidays.UnitedStates()

# Endpoint to get weekly sales predictions


@app.route('/api/weeklysaleslist', methods=['GET'])
def weekly_sales_list():
    next_7_days_predictions = []
    for i in range(1, 8):
        future_date = today + timedelta(days=i)
        future_day_of_week = future_date.weekday()
        future_is_weekend = calendar.day_name[future_day_of_week] in [
            'Saturday', 'Sunday']
        future_is_festive_day = future_date in us_holidays

        future_features = [future_day_of_week, future_is_weekend,
                           future_is_festive_day, future_date.month]
        future_sales_prediction = sales.model.predict([future_features])[0]
        next_7_days_predictions.append({"date": future_date.strftime(
            '%Y-%m-%d'), "prediction": future_sales_prediction})

    return jsonify({"weekly_sales_list": next_7_days_predictions})

# Endpoint to get today's sales prediction


@app.route('/api/todaysale', methods=['GET'])
def today_sale():
    today_day_of_week = today.weekday()
    today_is_weekend = calendar.day_name[today_day_of_week] in [
        'Saturday', 'Sunday']
    today_is_festive_day = today in us_holidays

    today_features = [today_day_of_week, today_is_weekend,
                      today_is_festive_day, today.month]
    today_sales_prediction = sales.model.predict([today_features])[0]
    return jsonify({"today_sale_prediction": today_sales_prediction})

# Endpoint to get total weekly sales prediction


@app.route('/api/totalweeksale', methods=['GET'])
def total_week_sale():
    total = 0
    for _, prediction in sales.next_7_days_predictions:
        total += prediction

    return jsonify({"total_week_sale_prediction": total})


# inventory

if __name__ == '__main__':
    app.run(debug=True)