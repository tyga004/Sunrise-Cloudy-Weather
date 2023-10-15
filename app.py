import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


def main():
    st.write("Group 4")
    st.write("Section: CPE 028 - CPE41S5")
    st.write("Instructor: Dr. Jonathan Taylar")
    st.title("Predicting Class Weather (Sunrise or Cloudy")
    st.write(
        "This program identifies submitted images according to their weather classification,"
        " whether they are Cloudy or Sunrise photos, using a pre-trained convolutional neural network model."
    )

    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model("weights-improvement-10-0.99.hdf5")
        return model

    def import_and_predict(image_data, model):
        image = np.asarray(image_data)
        image = image / 255.0
        img_reshape = np.reshape(image, (1, 128, 128, 3))
        prediction = model.predict(img_reshape)
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
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        string = "Prediction: " + class_name
        st.success(string)


if __name__ == "__main__":
    main()
