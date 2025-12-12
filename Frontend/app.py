import numpy as np
import pandas as pd
import streamlit as st
import joblib


@st.cache_resource
def load_model_and_features():
    model = joblib.load("airbnb_rf_model.joblib")
    feature_names = joblib.load("model_features.joblib")
    return model, feature_names


model, feature_names = load_model_and_features()

st.set_page_config(page_title="Airbnb Price Recommender", layout="centered")

st.title("Airbnb Price Recommender")
st.write(
    """
    This tool uses our tuned Random Forest model trained on Boston Airbnb data  
    to suggest a nightly price based on key listing characteristics.
    """
)

st.sidebar.header("Listing Details")

# USER INPUTS

neighbourhood = st.sidebar.text_input(
    "Neighborhood (e.g. Back Bay, Allston, Downtown)",
    value="Back Bay"
)

property_type = st.sidebar.selectbox(
    "Property type",
    [
        "Apartment",
        "House",
        "Condominium",
        "Loft",
        "Townhouse",
        "Guest suite",
        "Guesthouse",
        "Other"
    ]
)

room_type = st.sidebar.selectbox(
    "Room type",
    [
        "Entire home/apt",
        "Private room",
        "Shared room",
        "Hotel room"
    ]
)

accommodates = st.sidebar.slider("Accommodates (number of guests)", 1, 16, 4)
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 1)
bathrooms = st.sidebar.slider("Bathrooms", 0.0, 5.0, 1.0, step=0.5)

n_amenities = st.sidebar.slider("Number of amenities", 0, 60, 15)

host_is_superhost = st.sidebar.checkbox("Host is Superhost", value=True)

minimum_nights = st.sidebar.slider("Minimum nights", 1, 30, 2)
availability_365 = st.sidebar.slider("Available days per year", 0, 365, 200)

st.sidebar.markdown("---")
st.sidebar.write(
    "For any attributes not shown here, the model uses typical values learned "
    "from the training data."
)

# Start with all features set to NaN 
row = {col: np.nan for col in feature_names}

def set_if_present(col_name, value):
    if col_name in row:
        row[col_name] = value

set_if_present("neighbourhood_cleansed", neighbourhood)
set_if_present("property_type", property_type)
set_if_present("room_type", room_type)
set_if_present("accommodates", accommodates)
set_if_present("bedrooms", bedrooms)
set_if_present("bathrooms", bathrooms)
set_if_present("n_amenities", n_amenities)
set_if_present("host_is_superhost", host_is_superhost)
set_if_present("minimum_nights", minimum_nights)
set_if_present("availability_365", availability_365)

input_df = pd.DataFrame([row])

# PREDICTION

if st.button("Predict nightly price"):
    try:
        pred_price = model.predict(input_df)[0]

        st.subheader("Recommended Nightly Price")
        st.metric(label="Suggested price (USD)", value=f"${pred_price:,.2f}")

        st.write(
            """
            This recommendation is based on the tuned Random Forest model, which
            uses historical Boston Airbnb data and the listing details you provided.
            """
        )

    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")
else:
    st.info("Fill in the details on the left and click **Predict nightly price**.")