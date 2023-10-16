import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

def main():
    st.write("Group 4")
    st.write("Section: CPE 028 - CPE41S5")
    st.write("Instructor: Dr. Jonathan Taylar")
    st.title("Predicting Class Weather (Sunrise or Cloudy)")
    st.write(
        "This program identifies submitted images whether they are Cloudy or Sunrise photos."
    )

    login_success = False

    # Create input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Create a login button
    if st.button("Login"):
        if check_login(username, password):
            login_success = True
        else:
            st.error("Incorrect username or password")

    if login_success:
        run_prediction()

def run_prediction():
    # @st.cache(allow_output_mutation=True)
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

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        image = np.asarray(image)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        string = "Prediction: " + class_name
        st.success(string)

def check_login(username, password):
    # Replace with your authentication logic
    if username == "your_username" and password == "your_password":
        return True
    return False

if __name__ == "__main__":
    main()
