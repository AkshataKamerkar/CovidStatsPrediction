import streamlit as st
import joblib
import pandas as pd


# Load the dataset
data = pd.read_csv('covid_virus_dataset.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

look_back = 5

# App title
st.title("🦠 Covid Case Prediction 📊")
st.markdown("<hr>", unsafe_allow_html=True)
st.text("")


# Adding the main image and setting its style
image_path = "https://www.usda.gov/sites/default/files/covid-header-2.png"

st.markdown(
    f'<style>img {{ max-width: 100%; height: auto; max-height: 300px; }}</style>',
    unsafe_allow_html=True
)

st.image(image_path, caption="Covid Prediction", use_column_width=True)

# Taking Input from the User
state_name = st.text_input("Enter State Name:", "Maharashtra")
future_date = st.text_input("Enter Future Date (YYYY-MM-DD):", "2025-01-01")

# Converting future_date to datetime
future_date = pd.to_datetime(future_date)

# Loading the dataset
data = pd.read_csv('covid_virus_dataset.csv')  # Assuming you have the COVID dataset in this CSV

# Converting 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])


# Preparing the data for a specific state
state_data = data[data['States'] == state_name]
state_data = state_data[['Confirmed', 'Recovery', 'Deaths']]

# Loading the entire model using joblib
loaded_model = joblib.load('entire_model.joblib')

# Loading the scaler
scaler = joblib.load('scaler.pkl')

# Using the loaded model for prediction
last_data = state_data.tail(look_back)
scaled_input = scaler.transform(last_data[['Confirmed', 'Recovery', 'Deaths']])
X_pred = scaled_input.reshape((1, look_back, 3))

prediction = loaded_model.predict(X_pred)
prediction = scaler.inverse_transform(prediction)

# Displaying the 'Submit' Button
if st.button('Submit'):
    # Display prediction to the user
    st.text('')
    st.subheader("Prediction for {} on {}".format(state_name, future_date.strftime("%Y-%m-%d")))
    st.write("Confirmed Cases:", prediction[0][0])
    st.write("Recovery Cases:", prediction[0][1])
    st.write("Deaths:", prediction[0][2])

st.markdown("<hr>", unsafe_allow_html=True)

# Showing the Detailed Analysis
if st.button("Want to see Detailed Analysis?"):

    # Redirecting to the Tableau Dashboard
    st.markdown("[Open Dashboard](https://public.tableau.com/app/profile/sameer.kulkarni8181/viz/MinorProjectVisualization/Dashboard1)")
