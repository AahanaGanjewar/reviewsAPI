from flask import Flask, app, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import date, datetime, timedelta
import calendar
import holidays


app = Flask(__name__)

# Your provided data
sales_data = {
    # ... (your data here)

    "2020-12-18": 190,
    "2022-12-10": 1450,
    "2023-04-18": 7580,
    "2023-04-19": 14700,
    "2023-04-20": 13050,
    "2023-04-21": 4470,
    "2023-04-22": 11970,
    "2023-04-23": 10050,
    "2023-04-24": 13190,
    "2023-04-26": 8870,
    "2023-04-27": 1710,
    "2023-04-28": 2190,
    "2023-04-29": 300,
    "2023-04-30": 1430,
    "2023-05-04": 450,
    "2023-05-05": 820,
    "2023-05-07": 620,
    "2023-05-10": 540,
    "2023-05-11": 250,
    "2023-05-12": 200,
    "2023-05-13": 3000,
    "2023-05-14": 970,
    "2023-05-15": 1830,
    "2023-05-17": 2900,
    "2023-05-20": 1640,
    "2023-05-21": 230,
    "2023-05-22": 560,
    "2023-05-23": 80,
    "2023-05-24": 7990,
    "2023-05-25": 90,
    "2023-05-27": 470,
    "2023-05-30": 1450,
    "2023-05-31": 300,
    "2023-06-01": 2510,
    "2023-06-02": 280,
    "2023-06-09": 940,
    "2023-06-11": 420,
    "2023-06-19": 4270,
    "2023-06-20": 43740,
    "2023-06-21": 800,
    "2023-06-22": 2650,
    "2023-06-25": 13720,
    "2023-06-26": 5850,
    "2023-06-27": 3270,
    "2023-06-28": 2890,
    "2023-06-29": 480,
    "2023-06-30": 1280,
    "2023-07-01": 5040,
    "2023-07-02": 7000,
    "2023-07-03": 3140,
    "2023-07-04": 1280,
    "2023-07-05": 10170,
    "2023-07-06": 13690,
    "2023-07-07": 8840,
    "2023-07-08": 18840,
    "2023-07-09": 20940,
    "2023-07-10": 11440,
    "2023-07-11": 12980,
    "2023-07-12": 15600,
    "2023-07-13": 11480,
    "2023-07-14": 13150,
    "2023-07-15": 30080,
    "2023-07-16": 17570,
    "2023-07-17": 8510,
    "2023-07-18": 13050,
    "2023-07-19": 10030,
    "2023-07-20": 15780,
    "2023-07-21": 12480,
    "2023-07-22": 17300,
    "2023-07-23": 21450,
    "2023-07-24": 10550,
    "2023-07-25": 10880,
    "2023-07-28": 13410,
    "2023-07-29": 21640,
    "2023-07-30": 22260,
    "2023-07-31": 9560,
    "2023-08-01": 7550,
    "2023-08-02": 7990,
    "2023-08-03": 7590,
    "2023-08-04": 21340,
    "2023-08-05": 11070,
    "2023-08-06": 28140,
    "2023-08-07": 8110,
    "2023-08-08": 7660,
    "2023-08-09": 8080,
    "2023-08-10": 9390,
    "2023-08-14": 8990,
    "2023-08-15": 25040,
    "2023-08-16": 13470,
    "2023-08-17": 15220,
    "2023-08-18": 15070,
    "2023-08-19": 56070,
    "2023-08-20": 32030,
    "2023-08-21": 12200,
    "2023-08-22": 12000,
    "2023-08-23": 13900,
    "2023-08-24": 13180,
    "2023-08-25": 14820,
    "2023-08-26": 23600,
    "2023-08-27": 24770,
    "2023-08-28": 15330,
    "2023-08-29": 18990,
    "2023-08-30": 25130,
    "2023-08-31": 14920,
    "2023-09-01": 13770,
    "2023-09-02": 16740,
    "2023-09-03": 19650,
    "2023-09-04": 14120,
    "2023-09-05": 13310,
    "2023-09-06": 6490,
    "2023-09-08": 12200,
    "2023-09-09": 32910,
    "2023-09-10": 22240,
    "2023-09-11": 8750,
    "2023-09-12": 12780,
    "2023-09-13": 11850,
    "2023-09-14": 7390,
    "2023-09-15": 41900,
    "2023-09-16": 50120,
    "2023-09-17": 36640,
    "2023-09-18": 14450,
    "2023-09-19": 15450,
    "2023-09-20": 18310,
    "2023-09-21": 18580,
    "2023-09-22": 11100,
    "2023-09-23": 16560,
    "2023-09-24": 32610,
    "2023-09-25": 9870,
    "2023-09-26": 14170,
    "2023-09-27": 11500,
    "2023-09-29": 13210,
    "2023-09-30": 17750,
    "2023-10-01": 20530,
    "2023-10-02": 17830,
    "2023-10-03": 7260,
    "2023-10-04": 11540,
    "2023-10-05": 14550,
    "2023-10-06": 16640,
    "2023-10-07": 38720,
    "2023-10-08": 17130,
    "2023-10-09": 11240,
    "2023-10-10": 8710,
    "2023-10-11": 11910,
    "2023-10-12": 20940,
    "2023-10-13": 36340,
    "2023-10-14": 20010,
    "2023-10-15": 25750,
    "2023-10-16": 5370,
    "2023-10-17": 10610,
    "2023-10-18": 13430,
    "2023-10-19": 12700,
    "2023-10-20": 18490,
    "2023-10-21": 25990,
    "2023-10-22": 13900,
    "2023-10-23": 6830,
    "2023-10-24": 25380,
    "2023-10-25": 3230,
    "2023-10-26": 4450,
    "2023-10-27": 8200,
    "2023-10-28": 16040,
    "2023-10-29": 15400,
    "2023-10-30": 7470,
    "2023-10-31": 8070,
    "2023-11-01": 2840,
    "2023-11-02": 9120,
    "2023-11-03": 9240,
    "2023-11-04": 8820,
    "2023-11-05": 8040,
    "2023-11-06": 6770,
    "2023-11-07": 6840,
    "2023-11-08": 11570,
    "2023-11-09": 14875,
    "2023-11-10": 11050,
    "2023-11-11": 11810,
    "2023-11-12": 14020,
    "2023-11-13": 7340,
    "2023-11-14": 5990,
    "2023-11-15": 2870,
    "2023-11-16": 4470,
    "2023-11-17": 7720,
    "2023-11-18": 9040,
    "2023-11-19": 5490,
    "2023-11-20": 6030,
    "2023-11-21": 8660,
    "2023-11-22": 7880,
    "2023-11-23": 8990,
    "2023-11-24": 12300,
    "2023-11-25": 11670,
    "2023-11-26": 8820,
    "2023-11-27": 8250,
    "2023-11-28": 10330,
    "2023-11-29": 5690,
    "2023-11-30": 6940,
    "2023-12-01": 9030,
    "2023-12-02": 11460,
    "2023-12-03": 11960,
    "2023-12-04": 8680,
    "2023-12-05": 9930,
    "2023-12-06": 9870,
    "2023-12-07": 9970,
    "2023-12-08": 8910,
    "2023-12-09": 12320,
    "2023-12-10": 16730,
    "2023-12-11": 9360,
    "2023-12-12": 10610,
    "2023-12-13": 6820,
    "2023-12-14": 10640,
    "2023-12-15": 11780,
    "2023-12-16": 23160,
    "2023-12-17": 26000,
    "2023-12-18": 11270,
    "2023-12-19": 9760,
    "2023-12-20": 12450,
    "2023-12-21": 11970,
    "2023-12-22": 14380,
    "2023-12-23": 17740,
    "2023-12-24": 9320,
    "2023-12-25": 12890,
    "2023-12-26": 6260,
    "2023-12-27": 5900,
    "2023-12-29": 11120,
    "2023-12-30": 20760,
    "2023-12-31": 26130,
    "2024-01-01": 10950,
    "2024-01-02": 11270,
    "2024-01-03": 10000,
    "2024-01-04": 9940,
    "2024-01-05": 10190,
    "2024-01-06": 21950,
    "2024-01-07": 20000,
    "2024-01-08": 8280,
    "2024-01-09": 11540,
    "2024-01-10": 8700,
    "2024-01-11": 10650,
    "2024-01-12": 7540,
    "2024-01-13": 16190,
    "2024-01-14": 19610,
    "2024-01-15": 9900,
    "2024-01-16": 7240,
    "2024-01-17": 7430,
    "2024-01-19": 8790,
    "2024-01-20": 14830,
    "2024-01-21": 16360,
    "2024-01-22": 14850,
    "2024-01-23": 14010,
    "2024-01-24": 10800,
    "2024-01-25": 12360,
    "2024-01-26": 16530,
    "2024-01-27": 17030,
    "2024-01-28": 17960,
    "2024-01-29": 8670,
    "2024-01-30": 1480



}

# Convert date strings to datetime objects
sales_df = pd.DataFrame(list(sales_data.items()), columns=['Date', 'Sales'])
sales_df['Date'] = pd.to_datetime(sales_df['Date'])

# Identify weekends using the calendar library
sales_df['IsWeekend'] = sales_df['Date'].dt.day_name().isin(
    ['Saturday', 'Sunday'])

us_holidays = holidays.UnitedStates()
sales_df['IsFestiveDay'] = sales_df['Date'].apply(lambda x: x in us_holidays)

# Additional feature engineering (e.g., day of the week, month)
sales_df['DayOfWeek'] = sales_df['Date'].dt.dayofweek
sales_df['Month'] = sales_df['Date'].dt.month

# Train-Test Split
train_df, test_df = train_test_split(sales_df, test_size=0.2, shuffle=False)

# Feature selection
features = ['DayOfWeek', 'IsWeekend', 'IsFestiveDay', 'Month']

# Model Training
X_train = train_df[features]
y_train = train_df['Sales']

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Model Evaluation
X_test = test_df[features]
y_test = test_df['Sales']

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Prediction for today
today = datetime.now().date()
today_day_of_week = today.weekday()
today_is_weekend = calendar.day_name[today_day_of_week] in [
    'Saturday', 'Sunday']
today_is_festive_day = today in us_holidays

# Reshape the input for prediction to a 2D array
today_features = [[today_day_of_week, today_is_weekend,
                   today_is_festive_day, today.month]]

today_sales_prediction = model.predict(today_features)[0]
print(f'Predicted Sales for Today: {today_sales_prediction}')

# Predicted Sales for the next 7 days
next_7_days_predictions = []
for i in range(1, 8):
    future_date = today + timedelta(days=i)
    future_day_of_week = future_date.weekday()
    future_is_weekend = calendar.day_name[future_day_of_week] in [
        'Saturday', 'Sunday']
    future_is_festive_day = future_date in us_holidays

    future_features = [future_day_of_week, future_is_weekend,
                       future_is_festive_day, future_date.month]
    future_sales_prediction = model.predict([future_features])[0]
    next_7_days_predictions.append(
        (future_date.strftime('%Y-%m-%d'), future_sales_prediction))

total = 0
print("Predicted Sales for the Next 7 Days:")
for day, prediction in next_7_days_predictions:
    total += prediction
    print(f"{day}: {prediction}")

# Predicted Sales for the current week
current_week_predictions = model.predict(
    [[today_day_of_week, today_is_weekend, today_is_festive_day, today.month]])
print(f'Predicted Total Sales for the Current Week: {total}')

# # Predicted Sales for the current month
# current_month_predictions = model.predict([[today_day_of_week, today_is_weekend, today_is_festive_day, today.month]])
# print(f'Predicted Total Sales for the Current Month: {current_month_predictions}')


# Convert date strings to datetime objects
sales_df = pd.DataFrame(list(sales_data.items()), columns=['Date', 'Sales'])
sales_df['Date'] = pd.to_datetime(sales_df['Date'])

# Identify weekends using the calendar library
sales_df['IsWeekend'] = sales_df['Date'].dt.day_name().isin(
    ['Saturday', 'Sunday'])

# Identify festive days using the holidays library
us_holidays = holidays.UnitedStates()
sales_df['IsFestiveDay'] = sales_df['Date'].apply(lambda x: x in us_holidays)

# Additional feature engineering (e.g., day of the week, month)
sales_df['DayOfWeek'] = sales_df['Date'].dt.dayofweek
sales_df['Month'] = sales_df['Date'].dt.month

# Train-Test Split
train_df, test_df = train_test_split(sales_df, test_size=0.2, shuffle=False)

# Feature selection
features = ['DayOfWeek', 'IsWeekend', 'IsFestiveDay', 'Month']

# Model Training
X_train = train_df[features]
y_train = train_df['Sales']

model = RandomForestRegressor()
model.fit(X_train, y_train)
