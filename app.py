import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

def mobilenetv2_imagenet():
    st.title("Image Classification with MobileNetV2")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Convert image to RGB if it has an alpha channel
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        
        # Load MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        
        st.subheader("Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{label}: {score * 100:.2f}%")

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model", ("MobileNetV2 (ImageNet)",))
    
    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()

if __name__ == "__main__":
    main()
