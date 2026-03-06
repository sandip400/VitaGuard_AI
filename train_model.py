import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

features = [
    "HighBP", "HighChol", "BMI", "Smoker", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income"
]
X = df[features]
y = df["Diabetes_012"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully.")