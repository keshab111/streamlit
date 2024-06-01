import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Titanic Survival Prediction')

# Define user inputs
pclass = st.selectbox('Passenger Class', [1, 2, 3])
age = st.slider('Age', 0, 80, 29)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', 0, 10, 0)
parch = st.number_input('Number of Parents/Children Aboard', 0, 10, 0)
fare = st.number_input('Fare', 0, 500, 32)
sex = st.selectbox('Sex', ['male', 'female'])
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Sex_male': [1 if sex == 'male' else 0],
    'Embarked_Q': [1 if embarked == 'Q' else 0],
    'Embarked_S': [1 if embarked == 'S' else 0]
})

# Align input data with the model's expected input
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# Make prediction
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0][1]

# Display prediction
if prediction == 1:
    st.success(f'The passenger is likely to survive with a probability of {prediction_proba:.2f}.')
else:
    st.error(f'The passenger is not likely to survive with a probability of {prediction_proba:.2f}.')
