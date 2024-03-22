import streamlit as st
import numpy as np
from PIL import Image
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier

my_model = joblib.load("mnist_rf_model.joblib")

class Prediction:
    def __init__(self, border, my_model):
        self.border = border
        self.my_model = my_model
    
    def preprocess_image(self, border):
        img_resized = cv2.resize(border, (28, 28), interpolation=cv2.INTER_LINEAR)
        img_resized = cv2.bitwise_not(img_resized)
        return img_resized

    def predict(self):
        processed_image = self.preprocess_image(self.border)
        prediction = self.my_model.predict(processed_image.reshape(1, -1))
        return prediction

    def predict_proba(self):
        processed_image = self.preprocess_image(self.border)
        proba_prediction = self.my_model.predict_proba(processed_image.reshape(1, -1))
        proba_first_class = np.max(proba_prediction)
        return proba_first_class
    


def preprocess_image(original_image):
    found_contours = []
    if len(original_image.shape) > 2:
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = original_image
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        min_area = 100
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)

            inv = 255-thresh
            hPad = 0
            wPad = 0
    
            if h>w:
                wPad = int((h-w)/2)
            else:
                hPad = int((w-h)/2)
            border = cv2.copyMakeBorder(inv[y:y+h, x:x+w], hPad+20, hPad+20, wPad+20, wPad+20, cv2.BORDER_CONSTANT, value=(256))
            found_contours.append(border)
    return found_contours

def main():
    st.title("MNIST Digit Recognition")

    uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        pil_image = Image.open(uploaded_image)
        pil_image.thumbnail((400, 400))
        original_image = np.array(pil_image)

        processed_images = preprocess_image(original_image)

        st.image(original_image, caption='Original Image', use_column_width=True)

        for border in processed_images:
            prediction_instance = Prediction(border, my_model)

            prediction = prediction_instance.predict()
            proba_first_class = prediction_instance.predict_proba()

            col1, col2 = st.columns(2)

            with col1:
                st.image(border, caption='Preprocessed Image', use_column_width=True)

            img_resized = cv2.resize(border, (28, 28), interpolation=cv2.INTER_LINEAR)
            img_resized = cv2.bitwise_not(img_resized)
            
            with col2:
                st.image(img_resized, caption='Resized Image', use_column_width=True)

            st.write(f"Prediction: {prediction}")
            st.write(f"Probability of first class: {proba_first_class * 100: .2f}%")

if __name__ == "__main__":
    main()
