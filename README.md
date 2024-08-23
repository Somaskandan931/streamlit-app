import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import time


# Mock functions to simulate real-time data capture
def capture_eye_tracking_data():
    fixation_duration = np.random.uniform(200, 400)  # milliseconds (Typical fixation duration range)
    saccadic_amplitude = np.random.uniform(1, 5)  # degrees (Typical saccadic amplitude range)
    saccadic_velocity = np.random.uniform(100, 400)  # degrees/second (Typical saccadic velocity range)
    return fixation_duration, saccadic_amplitude, saccadic_velocity


def capture_speech_data():
    speech_rate = np.random.uniform(110, 160)  # words per minute (Typical speech rate for adults)
    pitch_variability = np.random.uniform(2, 4)  # Hz (Typical pitch variability range)
    return speech_rate, pitch_variability


# Load the pre-trained model and scaler
model = joblib.load(
    'C:/Users/somas/PycharmProjects/CreatingSyntheticDatasetsForEyeMovements/.venv/random_forest_model.pkl')
scaler = joblib.load('C:/Users/somas/PycharmProjects/CreatingSyntheticDatasetsForEyeMovements/.venv/scaler.pkl')


def main():
    st.title("Real-Time ADHD Likelihood Prediction While Reading")

    st.write("""
    ### Read the passage below. Real-time data will be captured to predict the likelihood of ADHD.
    """)

    passage = """Once upon a time in a faraway land, there lived a wise old owl. The owl was known throughout the forest for its wisdom and kindness. It spent its days watching over the animals and offering advice to those in need. One day, a young fox approached the owl, seeking guidance on how to find its way home. The owl, with a gentle hoot, pointed the fox in the right direction, and the young fox trotted off happily. The owl watched as the fox disappeared into the woods, knowing that it had helped another creature find its path."""

    st.text_area("Passage", value=passage, height=200, max_chars=None)

    # Capture control
    if "capturing" not in st.session_state:
        st.session_state.capturing = False

    if "latest_data" not in st.session_state:
        st.session_state.latest_data = None

    def start_capture():
        st.session_state.capturing = True

    def stop_capture():
        st.session_state.capturing = False

    if st.button("Start Capture"):
        start_capture()

    if st.button("Stop Capture"):
        stop_capture()

    if st.session_state.capturing:
        st.write("Capturing data... Please continue reading the passage above.")

        fixation_duration, saccadic_amplitude, saccadic_velocity = capture_eye_tracking_data()
        speech_rate, pitch_variability = capture_speech_data()

        # Store the latest captured data
        st.session_state.latest_data = {
            'Fixation_Duration': fixation_duration,
            'Saccadic_Amplitude': saccadic_amplitude,
            'Saccadic_Velocity': saccadic_velocity,
            'Speech_Rate': speech_rate,
            'Pitch_Variability': pitch_variability
        }

        time.sleep(1)  # Delay to mimic real-time data capture (in seconds)
        st.session_state.capturing = False  # Simulate single capture for demonstration

    if st.session_state.latest_data:
        # Display the captured data
        st.write(f"Fixation Duration: {st.session_state.latest_data['Fixation_Duration']:.2f} ms")
        st.write(f"Saccadic Amplitude: {st.session_state.latest_data['Saccadic_Amplitude']:.2f} degrees")
        st.write(f"Saccadic Velocity: {st.session_state.latest_data['Saccadic_Velocity']:.2f} degrees/second")
        st.write(f"Speech Rate: {st.session_state.latest_data['Speech_Rate']:.2f} words/min")
        st.write(f"Pitch Variability: {st.session_state.latest_data['Pitch_Variability']:.2f} Hz")

        # Prepare input data for prediction
        input_data = pd.DataFrame([st.session_state.latest_data])

        # Normalize the input data using the pre-fitted scaler
        scaled_input_data = scaler.transform(input_data)

        # Modify the threshold if necessary
        prediction_probability = model.predict_proba(scaled_input_data)[0, 1]
        threshold = 0.6  # Adjust based on model performance and validation
        prediction = 1 if prediction_probability > threshold else 0

        # Display the prediction and probability
        st.write(f"**Prediction: {'ADHD Likely' if prediction == 1 else 'ADHD Unlikely'}**")
        st.write(f"**Probability of ADHD: {prediction_probability:.2f}**")


if __name__ == "__main__":
    main()
