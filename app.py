import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Create a session state to store login status
if 'login_status' not in st.session_state:
    st.session_state.login_status = False

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_login(username, password):
            st.success("Login successful!")
            st.session_state.login_status = True  # Update login status
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
    if username == "user" and password == "user":
        return True
    return False

def main():
    st.title("Group 4")
    st.title("Section: CPE 028 - CPE41S5")
    st.title("Instructor: Dr. Jonathan Taylar")
    
    if not st.session_state.login_status:
        login()
    else:
        st.subheader("Welcome to the Prediction Page")
        run_prediction()

if __name__ == "__main__":
    main()
