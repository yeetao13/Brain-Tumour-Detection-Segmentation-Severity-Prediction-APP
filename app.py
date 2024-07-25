import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import keras.backend as K
import os
import joblib
import pandas as pd
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

VOLUME_SLICES = 100 
VOLUME_START_AT = 22

client = InferenceClient(
    os.getenv("MODEL"),
    token=os.getenv("TOKEN"),
)

def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss

def combined_loss(y_true, y_pred):
    dice = 1 - dice_coef(y_true, y_pred)
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return 0.5 * dice + 0.5 * cce

custom_objects = {'dice_coef': dice_coef, 'combined_loss': combined_loss}

# Load the saved model
classification_model = tf.keras.models.load_model('xception.h5')
segmentation_model = tf.keras.models.load_model('2d_unet_best_2.h5', custom_objects=custom_objects)

# Load the scaler and SVM model using joblib
scaler = joblib.load('scaler_before.pkl')
svm_model = joblib.load('svm_best.pkl')

def get_mask_volume(image_volume):
    totals = {1:0, 2:0, 3:0}
    for i in range(VOLUME_SLICES):
        arr=image_volume[:,:,i+VOLUME_START_AT].flatten()
        arr[arr == 4] = 3
        unique, counts = np.unique(arr, return_counts=True)
        unique = unique.astype(int)
        values_dict=dict(zip(unique, counts))
        for k in range(1,4):
            totals[k] += values_dict.get(k,0)
    return totals

def get_brain_volume(image_volume):
    total = 0
    for i in range(VOLUME_SLICES):
        arr=image_volume[:,:,i+VOLUME_START_AT].flatten()
        image_count=np.count_nonzero(arr)
        total=total+image_count
    return total

# Function to make classification
def classify(image):
    image = image.convert('RGB')
    img = image.resize((128, 128))
    
    img_array = img_to_array(img)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = classification_model.predict(img_array)
    predicted_class = 'class_1' if prediction[0][0] > 0.5 else 'class_0'
    confidence = prediction[0][0] if predicted_class == 'class_1' else 1 - prediction[0][0]
    return predicted_class, confidence

# Function to make segmentation
def predict_segmentation(flair_path, ce_path, start_slice=60):
    flair = nib.load(flair_path).get_fdata()
    ce = nib.load(ce_path).get_fdata()
    
    X = np.empty((VOLUME_SLICES, 128, 128, 2))

    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (128, 128))
        X[j, :, :, 1] = cv2.resize(ce[:, :, j + VOLUME_START_AT], (128, 128))
    
    p = segmentation_model.predict(X / np.max(X), verbose=1)

    st.write("Model prediction:")

    plt.figure(figsize=(18, 50))
    f, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(18, 50)) 

    ax1.imshow(cv2.resize(flair[:, :, start_slice + VOLUME_START_AT], (128, 128)), cmap="gray", interpolation='none')
    ax1.imshow(cv2.resize(flair[:, :, start_slice + VOLUME_START_AT], (128, 128)), cmap="gray")
    ax1.title.set_text('Image flair')

    ax2.imshow(cv2.resize(ce[:, :, start_slice + VOLUME_START_AT], (128, 128)), cmap="gray", interpolation='none')
    ax2.imshow(cv2.resize(ce[:, :, start_slice + VOLUME_START_AT], (128, 128)), cmap="gray")
    ax2.title.set_text('Image t1ce')

    ax3.imshow(cv2.resize(flair[:, :, start_slice + VOLUME_START_AT], (128, 128)), cmap="gray", interpolation='none')
    ax3.imshow(p[start_slice, :, :, 1:4], cmap="Reds", interpolation='none', alpha=0.3)
    ax3.title.set_text('Segmented result')

    
    st.pyplot(plt)

# Function to calculate mask and brain volumes and make prediction
def predict_lgg_hgg(flair_path, seg_path):
    flair_file = nib.load(flair_path)
    seg_file = nib.load(seg_path)

    masks = get_mask_volume(seg_file.get_fdata())
    brain_vol = get_brain_volume(flair_file.get_fdata())

    masks[1] = masks[1] / brain_vol
    masks[2] = masks[2] / brain_vol
    masks[3] = masks[3] / brain_vol

    combined = [masks[1], masks[2], masks[3]]
    data = [combined]

    df = pd.DataFrame(data, columns=['NECROTIC/CORE', 'EDEMA', 'ENHANCING'])

    df_scaled = scaler.transform(df)
    predictions = svm_model.predict(df_scaled)
    class_labels = {0: 'HGG', 1: 'LGG'}
    output = [class_labels[pred] for pred in predictions]
    return output[0]

st.title('Medical Image Analysis')

tab1, tab2, tab3 = st.tabs(["Classification", "Segmentation", "LGG/HGG Prediction"])

with tab1:
    st.header("Brain Tumor Classification")
    st.write("Upload an image to classify whether it has a brain tumor or not.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="class_upload")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        predicted_class, confidence = classify(image)

        if predicted_class == 'class_0':
            prompt = """
                You are an AI medical assistant specialized in brain tumor diagnosis and management.
                Your task is to assist healthcare professionals by providing detailed recommendations based on the results of a brain tumor classification system.
                The system predicts either no brain tumor is present or the presence of a brain tumor.
                List all the results in point form without specfying any name.

                The brain tumor classification system has predicted brain tumor not existed, indicating no brain tumor is present.
                In this scenario, reassure the patient that no brain tumor has been detected.
                Schedule regular follow-up visits to monitor the patient's condition and address any other symptoms the patient may have, considering other potential diagnoses.
                The neurologist should perform a thorough neurological examination to rule out other conditions and possibly conduct additional tests such as EEG or nerve conduction studies if neurological symptoms are present.
                Monitoring the patient periodically and advising on lifestyle modifications if needed is also recommended.

                The users should review and confirm the imaging results to ensure accuracy.
                They should recommend any follow-up imaging if there are ambiguous findings or if the patient's symptoms persist,
                and suggest alternative imaging techniques such as CT scans if necessary to rule out other pathologies.
                
                Generate different output each time and make it only 10 points, cannot more than 10.
                """
            st.write("Predicted Class: No Tumor")
            st.write(f"The model has determined that there is no brain tumor present in the uploaded image with a confidence of {confidence:.4f}.")
        else:
            prompt = """
                You are an AI medical assistant specialized in brain tumor diagnosis and management.
                Your task is to assist healthcare professionals by providing detailed recommendations based on the results of a brain tumor classification system.
                The system predicts either no brain tumor is present or the presence of a brain tumor.
                List all the results in point form without specfying any name.

                The brain tumor classification system has predicted brain tumor existed, indicating the presence of a brain tumor.
                In this scenario, it is essential to communicate the results to the patient with clarity and compassion.
                The healthcare team should then refer the patient to a neurologist and a radiologist for further evaluation.
                Immediate blood tests and a full physical examination should be arranged to assess the patient's general health.
                The neurologist's role would include ordering advanced diagnostic tests such as MRI with contrast, MRS, or PET scans to better understand the tumor's characteristics.
                Collaboration with oncologists and neurosurgeons would be necessary to devise a comprehensive treatment plan, which may consist of surgery, radiation, or chemotherapy based on the tumor's nature.

                The users should conduct detailed imaging studies, employing MRI with contrast to establish the tumor's size, location, and type.
                Utilization of advanced imaging techniques such as diffusion tensor imaging (DTI) or functional MRI (fMRI) is crucial to evaluate the influence of the tumor on nearby brain structures.
                A comprehensive report should be provided to the neurologist and oncologist to aid in treatment planning, including potential surgical options.
                The radiologist should also confirm the imaging results' accuracy, suggest any follow-up imaging if the patient's symptoms persist or if there are ambiguous findings,
                and recommend alternative imaging techniques if required to exclude other pathologies.
                
                Generate different output each time and make it only 10 points, cannot more than 10.
                """
            st.write("Predicted Class: Tumor Exists")
            st.write(f"The model has determined that there is a brain tumor present in the uploaded image with a confidence of {confidence:.4f}. Please consult a medical professional for further evaluation.")
        output=""
        for message in client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                stream=True,
            ):
            output += str(message.choices[0].delta.content)
        st.write(output)

with tab2:
    st.header("Brain Tumor Segmentation")
    st.write("Upload FLAIR and T1CE NIfTI files to get the segmentation mask for brain tumor.")

    flair_file = st.file_uploader("Choose the FLAIR NIfTI file...", type=["nii", "nii.gz"], key="flair_upload")
    ce_file = st.file_uploader("Choose the T1CE NIfTI file...", type=["nii", "nii.gz"], key="ce_upload")

    if flair_file is not None and ce_file is not None:
        with open("temp_flair.nii", "wb") as f:
            f.write(flair_file.getbuffer())
        with open("temp_ce.nii", "wb") as f:
            f.write(ce_file.getbuffer())

        # display images
        st.write("Input images:")
        test_image_flair=nib.load('temp_flair.nii').get_fdata()
        test_image_ce=nib.load('temp_ce.nii').get_fdata()
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20, 10))
        ax1.imshow(test_image_flair[:,:,90], cmap = 'gray')
        ax1.set_title('Image flair')
        ax2.imshow(test_image_ce[:,:,90], cmap = 'gray')
        ax2.set_title('Image t1ce')
        st.pyplot(plt)

        predict_segmentation("temp_flair.nii", "temp_ce.nii")

        os.remove("temp_flair.nii")
        os.remove("temp_ce.nii")

with tab3:
    st.header("LGG/HGG Prediction")
    st.write("Upload FLAIR and segmentation NIfTI files to predict LGG or HGG.")

    flair_file = st.file_uploader("Choose the FLAIR NIfTI file...", type=["nii", "nii.gz"], key="flair_upload_lgg_hgg")
    seg_file = st.file_uploader("Choose the segmentation NIfTI file...", type=["nii", "nii.gz"], key="seg_upload_lgg_hgg")

    if flair_file is not None and seg_file is not None:
        with open("temp_flair_lgg_hgg.nii", "wb") as f:
            f.write(flair_file.getbuffer())
        with open("temp_seg_lgg_hgg.nii", "wb") as f:
            f.write(seg_file.getbuffer())

        result = predict_lgg_hgg("temp_flair_lgg_hgg.nii", "temp_seg_lgg_hgg.nii")

        st.write(f"Predicted Class: {result}")
        if result == 'HGG':
            prompt = """
                You are an AI medical assistant specialized in brain tumor diagnosis and management.
                Your task is to assist healthcare professionals by providing detailed recommendations based on the results of a brain tumor severity classification system.
                The system predicts the severity of brain tumors, specifically distinguishing between high-grade glioma (HGG, Grade IV) and low-grade glioma (LGG, Grades I, II, and III).
                List all the results in point form without specifying any names.

                The brain tumor severity classification system has predicted HGG.
                Provide specific recommendations for managing this condition, focusing on treatment options, monitoring, and follow-up care.

                Example:
                - Discuss the results with the patient and their family, explaining the seriousness of the condition.
                - Recommend immediate referral to a neuro-oncologist for specialized care.
                - Arrange for a comprehensive MRI scan to further evaluate the tumor's characteristics.
                - Discuss potential treatment options, including surgery, radiotherapy, and chemotherapy.
                - Consider participation in clinical trials if available, as these may provide access to new therapies.
                - Begin planning for supportive care, including managing symptoms and maintaining quality of life.
                - Schedule regular follow-ups with the oncology team to monitor treatment response and manage side effects.
                - Evaluate the need for neurocognitive assessment to address any cognitive changes due to the tumor or treatment.
                - Coordinate with a multidisciplinary team, including neurosurgeons, radiation oncologists, and palliative care specialists.
                - Provide resources and support for the patient and their family, including counseling and support groups.

                This is an example. Generate different output each time with different combination and make it only 10 points, not more than 10.
                """
            st.write("High-Grade Glioma (HGG) is a malignant tumor consisting of Grade IV stage tumors. Please consult a medical professional for further evaluation and treatment.")
        else:
            prompt = """
                You are an AI medical assistant specialized in brain tumor diagnosis and management.
                Your task is to assist healthcare professionals by providing detailed recommendations based on the results of a brain tumor severity classification system.
                The system predicts the severity of brain tumors, specifically distinguishing between high-grade glioma (HGG, Grade IV) and low-grade glioma (LGG, Grades I, II, and III).
                List all the results in point form without specifying any names.

                The brain tumor severity classification system has predicted LGG.
                Provide specific recommendations for managing this condition, focusing on treatment options, monitoring, and follow-up care.

                Example:
                - Communicate the diagnosis clearly, emphasizing that LGGs are typically less aggressive.
                - Plan a detailed MRI to understand the tumor's extent and location.
                - Discuss potential treatment strategies, including observation, surgical resection, or radiotherapy, depending on the tumor's size, location, and symptoms.
                - Consider a biopsy to obtain a more accurate histological diagnosis if not already performed.
                - Schedule regular monitoring, including neuroimaging and clinical assessments, to detect any changes in the tumor.
                - Review the patient's neurological function regularly, with specific attention to potential changes in cognition or neurological status.
                - Discuss the possible long-term outcomes and need for ongoing surveillance.
                - Consider genetic and molecular testing of the tumor to tailor treatment strategies.
                - Encourage the patient to maintain a healthy lifestyle, including proper nutrition and physical activity, to support overall well-being.
                - Provide educational materials and support resources to help the patient and family understand the condition and treatment options.
                
                This is an example. Generate different output each time with different combination and make it only 10 points, not more than 10.
                """
            st.write("Low-Grade Glioma (LGG) is a benign tumor consisting of Grade I, II and III stage tumors. The cancer cells are absent in benign tumors with a homogenous structure. Regular monitoring via radiology is recommended.")
        output=""
        for message in client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            stream=True,
            ):
            output += str(message.choices[0].delta.content)
        st.write(output)

        os.remove("temp_flair_lgg_hgg.nii")
        os.remove("temp_seg_lgg_hgg.nii")