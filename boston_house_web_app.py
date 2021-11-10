import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression

st.write("""
# Boston House Price Predictior
##### Here, we predict the price of Boston House.
""")
st.write('---')

# Loads the Boston House Price Dataset
housing = datasets.load_boston()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
Y = pd.DataFrame(housing.target)

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
    DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features


# displaying the dataset
st.write("### Boston Dataset")
st.dataframe(X)
st.write('---')


df = user_input_features()

# Main Panel

# Print specified input parameters
st.write("### Specified Input Parameters")
st.dataframe(df)
st.write('---')

# splitting into training data and testing data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size=0.4, random_state=115)


# Build Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
# Apply Model to Make Prediction
prediction = model.predict(df)


# converting the predicted value from array to number
arr = np.array([[prediction]])
arr_zero = arr[0,0] 
arr_no = arr_zero.squeeze()

# displaying the actual test data
st.write("### Actual Test Data(Price)")
st.dataframe(Y)
st.write('---')

st.write("### Predicted Price of the New House based on the Input Parameters inserted by the User")
st.write(f"#### *Price :- $ {arr_no}*")
st.write('---')

# predicting accuracy of our model
accuracy = model.score(X_test,y_test)

st.write("### Accuracy of the Model")
st.write(f"#### *Accuracy :- {accuracy}*")
st.write('---')


# plotting the contributions of each parameter in graph
st.write("### Contributions of Each Input Parameter in Graph")
chart_data = pd.DataFrame(
    df,
    columns=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
)
st.bar_chart(chart_data)
st.write('---')


st.write("""
Created by ***Somenath Choudhury*** and ***Vishal Lazrus***
""")



