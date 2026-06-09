import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense

imp_data=pd.read_csv("./climate_change_indicators.csv")
imp_data = imp_data.fillna(0.0)
inital_year=1961
year_row=imp_data.iloc[:0,10:72]
yr_no=year_row.shape[1]

year_count=0
country_lst=[]
year_lst=[]
tempe_lst=[]

total_row=imp_data.shape[0]
for r in range(total_row):
    year_count=0
    curr_country=imp_data.loc[r,"Country"]
    for i in range(10,72):
        temp=imp_data.iloc[r,i]
        temp = float(temp)
        year=1961+year_count
        year_count=year_count+1
       
        country_lst.append(curr_country)
        year_lst.append(year)
        tempe_lst.append(temp)
        
df=pd.DataFrame({
    "Country":country_lst,
    "Year":year_lst,
    "temperature":tempe_lst
})

country_means = df.groupby('Country')['temperature'].mean()

df['Country_Target_Encoded'] = df['Country'].map(country_means)
df=df[['Country_Target_Encoded', 'Year', 'temperature']]

print(df)

# 1. Prepare your X and y from your processed DataFrame
X = df[['Country_Target_Encoded', 'Year',]]
y = df['temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model=tf.keras.models.Sequential()

model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16,activation="relu"))
model.add(Dense(units=1 ))

model.compile(optimizer="adam",loss='mse', metrics=['mae'])

model.fit(X_train_scaled, y_train,
    validation_split=0.15,
    epochs=50,       
    batch_size=32,    
    verbose=1)

country="India"
encoded_country=country_means[country]
import math
truncated_num = math.floor(encoded_country*10**8) / 10**8

raw_data = [[truncated_num, 1961]]


scaled_data = scaler.transform(raw_data)
sample_prediction = model.predict(scaled_data)

print(f"Predicted value: {sample_prediction[0][0]:.3f} °C")
