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


def run_prediction():
    st.title("Prediction")

    @st.cache(allow_output_mutation=True)
    def load_model():
        model = tf.keras.models.load_model("weights-improvement-10-0.99.hdf5")
        return model

    def import_and_predict(image_data, model):
        image = cv2.resize(image_data, (128, 128))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        return prediction

    model = load_model()
    class_names = ["CLOUDY", "SUNRISE"]

    file = st.file_uploader(
        "Choose a Cloudy or Sunrise picture from your computer",
        type=["jpg", "png", "jpeg"],
    )


    # Inside the run_prediction function
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        st.write(f"Image shape: {image.shape}")  # Add this line for debugging
        prediction = import_and_predict(image, model)
        # Add this line for debugging
        st.write(f"Prediction shape: {prediction.shape}")
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        string = "Prediction: " + class_name
        st.success(string)


def check_login(username, password):
    # Replace with your authentication logic
    if username == "user" and password == "user":
        return True
    return False


def main():
    st.title("Group 4")
    st.title("Section: CPE 028 - CPE41S5")
    st.title("Instructor: Dr. Jonathan Taylar")

    page = st.selectbox("Select Page", ["Login", "Prediction"])

    if page == "Login":
        if login():
            st.subheader("Welcome to the Prediction Page")
    elif page == "Prediction":
        run_prediction()


if __name__ == "__main__":
    main()
