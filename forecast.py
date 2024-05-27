import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import json


config = {
    "apiKey": "AIzaSyDqCK6IMgW2uvmKm4pF7KszXP2fdgT_NSA",
    "databaseURL": "https://onlyforimages-2e293-default-rtdb.asia-southeast1.firebasedatabase.app/"
}


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

# Create a DataFrame from the forecasts
forecast_df = pd.DataFrame(list(forecasts.items()), columns=['Item', 'Predicted Consumption'])

print(round(forecast_df))