import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets 
from sklearn.linear_model import LinearRegression

st.write("""
# California House Price Predictor
##### Here, we predict the price of California House.
""")
st.write('---')

# Loads the California House Price Dataset
housing = datasets.fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
Y = pd.DataFrame(housing.target, columns=["Price"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    MedInc = st.sidebar.slider('MedInc', float(X.MedInc.min()), float(X.MedInc.max()), float(X.MedInc.mean()))
    HouseAge = st.sidebar.slider('HouseAge', float(X.HouseAge.min()), float(X.HouseAge.max()), float(X.HouseAge.mean()))
    AveRooms = st.sidebar.slider('AveRooms', float(X.AveRooms.min()), float(X.AveRooms.max()), float(X.AveRooms.mean()))
    AveBedrms = st.sidebar.slider('AveBedrms', float(X.AveBedrms.min()), float(X.AveBedrms.max()), float(X.AveBedrms.mean()))
    Population = st.sidebar.slider('Population', float(X.Population.min()), float(X.Population.max()), float(X.Population.mean()))
    AveOccup = st.sidebar.slider('AveOccup', float(X.AveOccup.min()), float(X.AveOccup.max()), float(X.AveOccup.mean()))
    Latitude = st.sidebar.slider('Latitude', float(X.Latitude.min()), float(X.Latitude.max()), float(X.Latitude.mean()))
    Longitude = st.sidebar.slider('Longitude', float(X.Longitude.min()), float(X.Longitude.max()), float(X.Longitude.mean()))
    data = {'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup,
            'Latitude': Latitude,
            'Longitude': Longitude}
    features = pd.DataFrame(data, index=[0])
    return features

# displaying the dataset
st.write("### California Dataset")
st.dataframe(X)
st.write('---')

# displaying the details of input parameters
st.write("### Details of Input Parameters")
st.write("""
**MedInc**:- Median income in block group.

**HouseAge**:- Median house age in block group.

**AveRooms**:- Average number of rooms per household.

**AveBedrms**:- Average number of bedrooms per household.

**Population**:- Block group population.

**AveOccup**:- Average house occupancy.

**Latitude**:- Block group latitude.

**Longitude**:- Block group longitude.
""")
st.write("---")

df = user_input_features()

# Main Panel

# Print specified input parameters
st.write("### Specified Input Parameters")
st.dataframe(df)
st.write('---')

# splitting into training data and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=115)

# Build Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
# Apply Model to Make Prediction
prediction = model.predict(df)

# converting the predicted value from array to number
predicted_price = prediction[0, 0]

# displaying the actual test data
st.write("### Actual Test Data (Price)")
st.dataframe(Y)
st.write('---')

st.write("### Predicted Price of the New House based on the Input Parameters inserted by the User")
st.write(f"#### *Price :- $ {predicted_price:.2f}*")
st.write('---')

# predicting accuracy of our model
accuracy = model.score(X_test, y_test)

st.write("### Accuracy of the Model")
st.write(f"#### *Accuracy : {accuracy:.2f}*")
st.write('---')

# plotting the contributions of each parameter in graph
st.write("### Contributions of Each Input Parameter in the form of Bar Chart")
chart_data = pd.DataFrame(df)
st.bar_chart(chart_data)
st.write('---')

st.write("""
Created by ***Somenath Choudhury***
""")
