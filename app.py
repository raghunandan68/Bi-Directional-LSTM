import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Air Passenger Forecast",page_icon="✈️",layout="wide")

@st.cache_resource
def load_artifacts():
    model=load_model("airpassenger_birnn.keras",compile=False)
    scaler=joblib.load("airpassenger_scaler.pkl")
    return model,scaler

model,scaler=load_artifacts()

st.title("✈️ Air Passenger Forecasting Using Bidirectional RNN")

tab1,tab2=st.tabs(["Forecast","Model Information"])

with tab1:

    uploaded_file=st.file_uploader("Upload Dataset",type=["csv"])

    if uploaded_file is not None:

        df=pd.read_csv(uploaded_file)

        st.dataframe(df.head())

        if "#Passengers" not in df.columns:

            st.error("Dataset must contain #Passengers column")

        else:

            passengers=df["#Passengers"].values.reshape(-1,1)

            if len(passengers)<12:

                st.error("Dataset must contain at least 12 rows")

            else:

                scaled_passengers=scaler.transform(passengers)

                if st.button("Forecast Next 7 Months"):

                    last_12_months=scaled_passengers[-12:].flatten().tolist()

                    future_passengers=[]

                    for _ in range(7):
                        x=np.array(last_12_months[-12:]).reshape(1,12,1)
                        pred=model.predict(x,verbose=0)[0][0]
                        future_passengers.append(pred)
                        last_12_months.append(pred)

                    future_passengers=scaler.inverse_transform(np.array(future_passengers).reshape(-1,1))

                    forecast_df=pd.DataFrame({
                        "Month":[f"Month {i}" for i in range(1,8)],
                        "Predicted Passengers":future_passengers.flatten().astype(int)
                    })

                    st.subheader("Next 7 Months Forecast")

                    st.dataframe(forecast_df,use_container_width=True)

                    st.line_chart(forecast_df.set_index("Month"))

with tab2:

    st.markdown("""
    ### Bidirectional RNN Architecture

    - Bidirectional SimpleRNN (64 Units)
    - Dropout (0.2)
    - Bidirectional SimpleRNN (32 Units)
    - Dropout (0.2)
    - Dense (16 Units)
    - Dense (1 Unit)

    ### Input
    Previous 12 Months Passenger Data

    ### Output
    Next 7 Months Passenger Forecast

    ### Algorithm
    Bidirectional Recurrent Neural Network (BiRNN)

    ### Problem Type
    Time Series Forecasting
    """)