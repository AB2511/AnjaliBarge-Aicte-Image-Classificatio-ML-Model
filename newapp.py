import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

def cifar10_classification():
    st.title("CIFAR-10 Image Classification")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Convert image to RGB if it has an alpha channel
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        
        # Load CIFAR-10 model
        model = tf.keras.models.load_model('model111.keras')

        
        # Compile the model (important for evaluation)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Preprocess the image
        img = image.resize((32, 32))  # CIFAR-10 image size
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Class with highest probability
        confidence = np.max(predictions)  # Max probability for the class
        
        st.subheader("Predictions:")
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
        
        # Display the prediction probabilities in a bar chart
        st.subheader("Class Probabilities:")
        st.bar_chart(predictions.flatten())  # Display the flattened array as a bar chart
        
        # Optionally, you can add labels to the bars
        st.write("Class Probabilities for Each Class:")
        st.write(dict(zip(class_names, predictions.flatten())))

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model", ("CIFAR-10",))
    
    if choice == "CIFAR-10":
        cifar10_classification()

if __name__ == "__main__":
    main()
