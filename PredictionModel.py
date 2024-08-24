import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load your synthetic datasets
eye_tracking_data = pd.read_csv("C:/Users/somas/PycharmProjects/CaptureRealTimeData/synthetic_adhd_dataset.csv")
speech_data = pd.read_csv("C:/Users/somas/PycharmProjects/CaptureRealTimeData/synthetic_speech_dataset.csv")

# Merge datasets on Participant_ID
combined_data = pd.merge(eye_tracking_data, speech_data, on='Participant_ID')

# Features and target variable
X = combined_data[['Fixation_Duration', 'Saccadic_Amplitude', 'Saccadic_Velocity', 'Speech_Rate', 'Pitch_Variability']]
y = combined_data['Label_x']  # Assuming Label_x is your target variable for ADHD prediction

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model (optional, for your own analysis)
y_pred = model.predict(X_test_scaled)
accuracy = (y_pred == y_test).mean()
print(f"Model accuracy: {accuracy:.2f}")

# Save the trained model and scaler
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
