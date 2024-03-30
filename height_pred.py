import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import accuracy_score

# Step 1: Upload data and clean
df = pd.read_csv("new_fat_dataset.csv")
pd.set_option('future.no_silent_downcasting', True)
arr = set(list(df["NObeyesdad"]))
name_change = {
        "Normal_Weight" : 0,
        "Insufficient_Weight": 1,
        "Overweight_Level_I": 2,
        "Overweight_Level_II": 3,
        "Obesity_Type_I": 4,
        'Obesity_Type_II': 5,
        "Obesity_Type_III": 6
}
df["Obesity_Level"] = df["NObeyesdad"].replace(name_change)
df["BMI"] = df["Weight"] / df["Height"]


# Step 2: Train the model

X = df[["Weight"]].values.reshape(-1,1)
y = np.array(df["Height"])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=10)
reg = LinearRegression()
reg.fit(X_train,y_train)
pred = reg.predict(X_test)
# print(f'Your height has been predicted with the accuracy of {accuracy}')
sns.regplot(x=X_train, y=y_train, data=df)
# plt.title("Weights and Heights prediction through regression line")
# plt.show()
# plt.pause(1)
# plt.close()


# Step 3: Create an user interface for filling the weight and height prediction

st.title(
        '''
        Welcome to Height Prediction
        '''
)
st.caption('Your height will be predicted')
st.image("https://cdn5.vectorstock.com/i/1000x1000/56/24/cute-smiling-happy-strong-avocado-make-gym-vector-26755624.jpg")
weight = st.number_input('Please enter your current weight (in kilograms):')
weight_arr = [[weight]]

st.write(f"Your predicted height is {reg.predict(weight_arr)} meters")