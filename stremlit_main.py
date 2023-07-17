import streamlit as st
import joblib
import pandas as pd
# import xgboost as xgb
from PIL import Image




st.set_page_config(
    page_title='Car Price Predictor',
    page_icon='icon_car.png'
)

st.header("Car Price Prediction")

image = Image.open('car_img.jpg')

st.image(image, caption='A red car', width=300)


# Year
year = st.number_input("Year", min_value=2014, max_value=2023, value=2019, step=1)

#  Present_Price
price = st.slider("Price", min_value=1000, max_value=100_000, value=3000, step=250)


#  Kms_Driven
kms = st.slider('Kms Driven', min_value=0, max_value=300_000, value=3000, step=1)

#  Owner
owner = st.selectbox('How many owners in order?',
    (0, 1, 3))


#  Fuel_Type
fuel_type = st.radio('Fuel Type:', ('Diesel', 'Petrol'))

#  Seller_Type
seller_type = st.radio('Seller Type:', ('Individual', 'Dealer'))

#  Transmission
transmission = st.radio('Transmission Type:', ('Manual', 'Automatic'))


columns = joblib.load("features_list.joblib")


user_input = [{
"Year":                     year,
"Present_Price":            price / 10_000,
"Kms_Driven":               kms,
"Fuel_Type":                fuel_type,
"Seller_Type":              seller_type,
"Transmission":             transmission,
"Owner":                    owner
    }]


df_s = pd.DataFrame(user_input)

df_s["Year"] = 2023-df_s["Year"]
df_s = pd.get_dummies(df_s).reindex(columns=columns, fill_value=0)

# st.write('Columns from file:')
# st.write(columns)

# st.write('df_s:')
# st.write(df_s.columns)


#load model and scaler
scaler = joblib.load(open("scaler.joblib","rb"))
model = joblib.load(open("xgb_model.joblib","rb"))


df_s = scaler.transform(df_s)

btn_predict = st.button('Predict Price')

if btn_predict:
    pred_price = round(model.predict(df_s)[0] * 10_000)
    st.write(f"Your car's price: ${pred_price}")




