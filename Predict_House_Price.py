import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from  sklearn.metrics import mean_squared_error
from  sklearn.linear_model import LinearRegression
import plotly.express as px 

def filter_data_size(size):
    size = size[size> 0]
    size = pd.Series(size)
    Q1 = size.quantile(0.25)
    Q3 = size.quantile(0.75)
    IQR = Q3 -Q1 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 +1.5 * IQR
    filtered = size[(size > lower_bound) & (size < upper_bound)]
    return filtered

def generate_house_data (n_samples =100):
    np.random.seed(50)
    size = np.random.normal(75 , 60 , n_samples)
    size =  filter_data_size(size)
    price = size * 84530000 + np.random.normal(0 , 50 ,len(size))
    return pd.DataFrame({"size": size , "price": price })

def train_model():
    df = generate_house_data()
    x = df[["size"]]
    y = df[["price"]]
    x_train , x_test ,  y_train ,  y_test = train_test_split(x , y , test_size=0.2)

    model = LinearRegression()
    model.fit(x_train , y_train)

    return model

def main():
    st.title("Simple Linear Regression House Prediction App :)")
    st.write("put in your house size to know its price.....")

    model = train_model()

    size = st.number_input("House Size", min_value=20, max_value=300, value=75)

    if st.button("predict price"): 
        predicted_Price = model.predict([[size]])
        st.success(f"Estimate price : تومان  {float(predicted_Price[0]):,.0f}")

        df = generate_house_data()

        fig = px.scatter(df, x="size", y="price", title="Size VS House Price")
        fig.add_scatter(
            x=[size],
            y=[float(predicted_Price[0])],
            mode="markers",
            marker=dict(size=15, color="red"),
            name="Prediction"
        )

        st.plotly_chart(fig)



if __name__ == "__main__" : 
    main()
    


