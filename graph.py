import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file into a Pandas DataFrame
data = pd.read_excel("healthcare_data.xlsx")

# One-hot encode the categorical columns
data = pd.get_dummies(data, columns=['Gender', 'Cholesterol', 'Smoking', 'Family History', 'Exercise'],
                      prefix=['Gender', 'Cholesterol', 'Smoking', 'Family History', 'Exercise'])

# Define the features (X) and target variable (y)
X = data[['Age', 'Systolic', 'Diastolic', 'Gender_Female', 'Gender_Male', 'Cholesterol_High', 'Cholesterol_Normal',
          'Smoking_No', 'Smoking_Yes', 'Family History_No', 'Family History_Yes', 'Exercise_No', 'Exercise_Yes']]
y_heart_disease = data['Heart Disease']

# Convert 'No' and 'Yes' to 0 and 1
y_heart_disease = y_heart_disease.map({'No': 0, 'Yes': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_heart_disease, test_size=0.2, random_state=42)

# Create and train the KNN model
knn_heart_disease = KNeighborsClassifier(n_neighbors=5)
knn_heart_disease.fit(X_train, y_train)

# Create and train the Decision Tree model
dt_heart_disease = DecisionTreeClassifier(random_state=42)
dt_heart_disease.fit(X_train, y_train)

# Create and train the XGBoost model
xgb_heart_disease = XGBClassifier(random_state=42)
xgb_heart_disease.fit(X_train, y_train)

# Check the number of features in your training data
num_features = X_train.shape[1]

# Create and train the ANN model
ann_heart_disease = Sequential()
ann_heart_disease.add(Dense(8, input_dim=num_features, activation='relu'))
ann_heart_disease.add(Dense(1, activation='sigmoid'))
ann_heart_disease.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
ann_heart_disease.fit(X_train, y_train, epochs=10, verbose=0)

# Evaluate performance on the test set
models = [knn_heart_disease, dt_heart_disease, xgb_heart_disease, ann_heart_disease]
model_names = ['KNN', 'Decision Tree', 'XGBoost', 'ANN']

# Initialize lists to store performance metrics
accuracies, precisions, recalls, f1_scores = [], [], [], []

for model in models:
    y_pred = model.predict(X_test)
    if isinstance(model, Sequential):  # Check if the model is ANN
        y_pred = (y_pred > 0.5).astype(int).reshape(-1)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

# Plotting the performance metrics with color-coded graphs
plt.figure(figsize=(12, 8))

# Plotting Accuracy
plt.subplot(2,2, 1)
sns.barplot(x=model_names, y=accuracies, palette=['#494623', '#c3892b', '#8e883d', '#8f7c2f'])
plt.title('Accuracy')

# Plotting Precision
plt.subplot(2, 2, 2)
sns.barplot(x=model_names, y=precisions, palette=['#494623', '#c3892b', '#8e883d', '#8f7c2f'])
plt.title('Precision')

# Plotting Recall
plt.subplot(2, 2, 3)
sns.barplot(x=model_names, y=recalls, palette=['#494623', '#c3892b', '#8e883d', '#8f7c2f'])
plt.title('Recall')

# Plotting F1-Score
plt.subplot(2, 2, 4)
sns.barplot(x=model_names, y=f1_scores, palette=['#494623', '#c3892b', '#8e883d', '#8f7c2f'])
plt.title('F1-Score')

plt.tight_layout()
plt.show()

# Function to get user input and preprocess it
def get_user_input():
    print("Please enter the following details:")
    age = int(input("Age: "))
    systolic = int(input("Systolic blood pressure: "))
    diastolic = int(input("Diastolic blood pressure: "))
    gender = input("Gender (Male/Female): ")
    cholesterol = input("Cholesterol level (Normal/High): ")
    smoking = input("Do you smoke? (Yes/No): ")
    family_history = input("Family history of heart disease? (Yes/No): ")
    exercise = input("Do you exercise regularly? (Yes/No): ")

    # Creating a DataFrame from user input
    input_data = pd.DataFrame([[age, systolic, diastolic, gender, cholesterol, smoking, family_history, exercise]],
                              columns=['Age', 'Systolic', 'Diastolic', 'Gender', 'Cholesterol', 'Smoking', 'Family History', 'Exercise'])
    return input_data

# Function to preprocess input data
def preprocess_input(input_data):
    # One-hot encode the categorical columns
    input_data = pd.get_dummies(input_data, columns=['Gender', 'Cholesterol', 'Smoking', 'Family History', 'Exercise'],
                                prefix=['Gender', 'Cholesterol', 'Smoking', 'Family History', 'Exercise'])
    # Ensure all columns from the original dataset are present
    for col in X_train.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    return input_data

# Get user input and preprocess
user_input = get_user_input()
preprocessed_input = preprocess_input(user_input)

# Making predictions with each model
print("\nPredictions:")
for name, model in zip(model_names, models):
    if isinstance(model, Sequential):  # Check if the model is ANN
        prediction = model.predict(preprocessed_input)
        prediction = (prediction > 0.5).astype(int)[0][0]
    else:
        prediction = model.predict(preprocessed_input)[0]

    result = 'Positive' if prediction == 1 else 'Negative'
    print(f"{name} model prediction for heart disease: {result}")
