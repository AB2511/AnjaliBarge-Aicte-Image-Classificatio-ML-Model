AnjaliBarge-Aicte-Image-Classificatio-ML-Model


This project provides an interactive image classification web app built with **Streamlit**. It classifies images using a **CIFAR-10** dataset model and **MobileNetV2** trained on ImageNet. The app allows users to upload images and get real-time predictions.

## Features
- **CIFAR-10 Image Classification**: Classifies images into 10 categories from the CIFAR-10 dataset, including categories like airplane, bird, cat, dog, etc.
- **MobileNetV2 Classification**: Classifies images using the MobileNetV2 model trained on ImageNet.
- **Streamlit Interface**: A simple and interactive web interface that allows image upload, display, and prediction.
- 
## File Structure
AnjaliBarge-Aicte-Image-Classificatio-ML-Model/
```
├── train_model.py        # Script to train and save the CIFAR-10 model
├── newapp.py             # Streamlit app for CIFAR-10 classification
├── app.py                # Streamlit app for MobileNetV2 classification
├── model111.keras        # Pre-trained CIFAR-10 model (saved in Keras format)
├── requirements.txt      # Dependencies required for the project
└── README.md             # Project documentation
```

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/AnjaliBarge/AnjaliBarge-Aicte-Image-Classificatio-ML-Model.git

### 2. Create a Virtual Environment (optional but recommended)
```bash
python -m venv env
```

### 3. Activate the Virtual Environment
- **Windows**:
  ```bash
  env\Scripts\activate
  ```
- **Mac/Linux**:
  ```bash
  source env/bin/activate
  ```

### 4. Install Dependencies
Install the required packages listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 5. Train the CIFAR-10 Model (if necessary)
If you do not have the pre-trained model `model111.keras`, you can train it using the `train_model.py` script.

Run the script to train the model:
```bash
python train_model.py
```

The trained model will be saved as `model111.keras` in the project directory.

### 6. Run the Streamlit App
You can now run either of the two Streamlit apps:

- For **MobileNetV2 Image Classification**:
  ```bash
  streamlit run app.py
  ```

- For **CIFAR-10 Image Classification**:
  ```bash
  streamlit run newapp.py
  ```

After running the command, the app will be accessible in your browser at `http://localhost:8501`.

## How to Use

1. **Upload an image**: Use the file uploader widget to upload an image (JPEG/PNG).
2. **Classify Image**: Once the image is uploaded, the app will classify the image and show the predicted class along with the confidence score.
3. **Class Probabilities**: For CIFAR-10, the app will also display a bar chart showing the probabilities of all classes.

## Models Used

1. **CIFAR-10 Model**:
   - The model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes.
   - The model is a simple Convolutional Neural Network (CNN) that was trained for 10 epochs.

2. **MobileNetV2 Model**:
   - The model is pre-trained on the ImageNet dataset, a large visual database used for machine learning.
   - MobileNetV2 is a lightweight model designed for mobile and edge devices, balancing performance and efficiency.

## Dependencies

This project requires the following libraries:
- `streamlit` for the web interface
- `tensorflow` for machine learning and model inference
- `PIL` for image processing
- `numpy` for numerical operations

The `requirements.txt` file includes all the dependencies you need to run the project.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- **TensorFlow** for providing powerful tools for building machine learning models.
- **Streamlit** for enabling rapid development of interactive web applications.
- **CIFAR-10 Dataset** and **ImageNet Dataset** for their well-known image classification datasets.
