import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Load the dataset
data = pd.read_csv('covid_virus_dataset.csv')  # Assuming you have the COVID dataset in this CSV

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Prepare the data for a specific state
def predict_future_covid_stats(state_name, future_date):
    state_data = data[data['States'] == state_name]

    # Select relevant columns and scale the data
    state_data = state_data[['Date', 'Confirmed', 'Recovery', 'Deaths']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(state_data[['Confirmed', 'Recovery', 'Deaths']])

    # Define the look-back period
    look_back = 5

    # Prepare sequences for the LSTM model
    def prepare_sequences(data, look_back=5):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back)])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)

    X, y = prepare_sequences(scaled_data, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 3))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 3)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(3))  # 3 outputs for Confirmed, Recovery, Deaths

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model
    model.fit(X, y, epochs=50, batch_size=32)

    # Prepare data for prediction on the future date
    last_data = state_data.tail(look_back)
    scaled_input = scaler.transform(last_data[['Confirmed', 'Recovery', 'Deaths']])
    X_pred = scaled_input.reshape((1, look_back, 3))

    # Predict for the future date
    prediction = model.predict(X_pred)
    prediction = scaler.inverse_transform(prediction)

    # Create a DataFrame with the predicted values
    future_predicted_data = pd.DataFrame({
        'State': [state_name],
        'Date': [future_date],
        'Confirmed': [prediction[0][0]],
        'Recovery': [prediction[0][1]],
        'Deaths': [prediction[0][2]]
    })

    model.save_weights('model_weights.h5')
    joblib.dump(model, 'entire_model.joblib')

    joblib.dump(scaler, 'scaler.pkl')

    return future_predicted_data

# Input the future date and state
future_date = pd.to_datetime('2024-1-01')
state_name = 'Karnataka'

# Predict future COVID statistics for the given state on the specified date
predicted_stats = predict_future_covid_stats(state_name, future_date)
print(predicted_stats)



