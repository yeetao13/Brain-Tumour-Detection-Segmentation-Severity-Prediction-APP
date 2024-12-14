# FYP - Brain Tumour Detection, Segmentation & Severity Prediction

This project is designed to detect, segment, and predict the severity of brain tumors using a Streamlit web application. The application leverages deep learning & machine learning models to analyze MRI scans and provides a user-friendly interface for interacting with the model.

## Features

- **Brain Tumor Detection:** Identify the presence of a brain tumor from MRI images.
- **Tumor Segmentation:** Precisely segment the detected tumor within the MRI image.
- **Severity Prediction:** Predict the severity level of the detected tumor based on its characteristics.
  
## Installation

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/yeetao13/Brain-Tumour-Detection-Segmentation-Severity-Prediction-APP.git
cd Brain-Tumour-Detection-Segmentation-Severity-Prediction-APP
```
### 2. Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.
Python must be version 310 (Python 3.10.2) to run the Streamlit web application

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Once the virtual environment is activated, install the required packages by running:

```bash
pip install -r requirements.txt
```

### 4. Create a .env File

You will need to create a .env file in the root directory of the project to store your Hugging Face API token and model names. The .env file should look like this:

```bash
TOKEN=your_huggingface_token
MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
```

### 5. Run the Application

After setting up your environment and .env file, you can run the Streamlit app using the following command:

```bash
streamlit run app.py
```

Note:
- `xception.h5` file is not included in the repo as the file size is too large
- Create a virtual environment and install python with version 310
- python must be version 310 (Python 3.10.2) to run the Streamlit web application
- 2d_unet_best_2.h5: The best segmentation model
- xception.h5: The best classification model
- svm_best.pkl: The best severity prediction model
- scaler_before.pkl: The scaler of SVM model (Before SMOTE)
- app.py: The source code of web application
- requirements.txt: The required libraries to run the web application
- Training scripts are not provided in this repo
