import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load healthcare data from an Excel file
data_set = pd.read_excel('healthcare_data.xlsx')


# Update column indices based on the actual columns in your dataset
# Assuming 'AGE', 'GENDER', 'SYSTOLIC', 'DIASTOLIC', 'CHOLESTEROL', 'SMOKING', 'FAMILY_HISTORY', 'EXERCISE', 'EXANG' are columns in your dataset
features = ['AGE', 'GENDER', 'SYSTOLIC', 'DIASTOLIC', 'CHOLESTEROL', 'SMOKING', 'FAMILY_HISTORY', 'EXERCISE', 'EXANG']

# Prompt user for input
user_input = {}
for feature in features:
    user_input[feature] = float(input(f"Enter value for {feature}: "))

# Create a DataFrame with user input
user_df = pd.DataFrame([user_input])

# Combine user input with the original dataset
combined_data = pd.concat([data_set[features], user_df])

# Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(combined_data)

# Separate user input from the combined data
user_input_scaled = scaled_data[-1:]

# Separate the dataset
x = scaled_data[:-1]
y = data_set['TARGET_COLUMN'].values  # Update 'TARGET_COLUMN' with the actual column name for the target variable

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Choose a classification algorithm (e.g., Logistic Regression)
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# Predictions for user input
user_pred = classifier.predict(user_input_scaled)
print(f"Prediction for user input: {user_pred[0]}")
