import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

def login():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "user" and password == "user":
            return True
        else:
            st.warning("Incorrect username or password")
    return False

def main():
    st.write("Group 4")
    st.write("Section: CPE 028 - CPE41S5")
    st.write("Instructor: Dr. Jonathan Taylar")
    st.title("Predicting Class Weather (Sunrise or Cloudy)")
    st.write(
        "This program identifies submitted images whether they are Cloudy or Sunrise photos."
    )

    if login():
        st.text_input("Username", value="user", key="username_input")  # Hide username input
        st.text_input("Password", type="password", value="user", key="password_input")  # Hide password input

        uploaded = st.file_uploader(
            "Choose a Cloudy or Sunrise picture from your computer",
            type=["jpg", "png", "jpeg"],
        )

        if uploaded is None:
            st.text("Please upload an image file")
        else:
            image = Image.open(uploaded)
            image = np.asarray(image)
            st.image(image, use_column_width=True)
            with st.spinner("Analyzing..."):
                model = tf.keras.models.load_model("weights-improvement-10-0.99.hdf5")
                class_names = ["CLOUDY", "SUNRISE"]
                image_data = cv2.resize(image, (128, 128))
                image_data = image_data / 255.0
                image_data = np.expand_dims(image_data, axis=0)
                prediction = model.predict(image_data)
                class_index = np.argmax(prediction)
                class_name = class_names[class_index]
                string = "Prediction: " + class_name
                st.success(string)

if __name__ == "__main__":
    main()
