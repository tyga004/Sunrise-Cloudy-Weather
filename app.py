import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_login(username, password):
            st.success("Login successful!")
            return True
        else:
            st.error("Incorrect username or password")
    return False

def run_prediction(model):
    st.title("Prediction")
    class_names = ["CLOUDY", "SUNRISE"]

    uploaded_image = st.file_uploader(
        "Choose a Cloudy or Sunrise picture from your computer",
        type=["jpg", "png", "jpeg"],
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)
        image_data = np.asarray(image)
        st.write(f"Image shape: {image_data.shape}")
        
        prediction = import_and_predict(image_data, model)
        st.write(f"Prediction shape: {prediction.shape}")

        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        st.success("Prediction: " + class_name)

def import_and_predict(image_data, model):
    image = cv2.resize(image_data, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

def check_login(username, password):
    # Replace with your authentication logic
    if username == "user" and password == "user":
        return True
    return False

def main():
    st.title("Group 4")
    st.title("Section: CPE 028 - CPE41S5")
    st.title("Instructor: Dr. Jonathan Taylar")

    logged_in = False
    model = tf.keras.models.load_model("weights-improvement-10-0.99.hdf5")

    page = st.selectbox("Select Page", ["Login", "Prediction"])

    if page == "Login":
        logged_in = login()
        if logged_in:
            st.subheader("Welcome to the Prediction Page")
    elif page == "Prediction" and logged_in:
        run_prediction(model)

if __name__ == "__main__":
    main()
