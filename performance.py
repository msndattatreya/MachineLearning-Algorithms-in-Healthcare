import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from tensorflow import keras
from sklearn.metrics import accuracy_score
from tabulate import tabulate

# Load the Excel file into a Pandas DataFrame
data = pd.read_excel("healthcare_data.xlsx")

# One-hot encode categorical columns
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

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Train Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Train XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Train ANN model
ann_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(1, activation='sigmoid')
])
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train, epochs=10, verbose=0)

# Take user input
user_input = {}
for feature in X.columns:
    user_input[feature] = input(f"Enter {feature}: ")

# Prepare the user input data for prediction
user_input = pd.DataFrame([user_input])

# One-hot encode the user's input
user_input = pd.get_dummies(user_input, columns=X.columns)

# Predictions for each model
knn_prediction = knn_model.predict(user_input)
dt_prediction = dt_model.predict(user_input)
xgb_prediction = xgb_model.predict(user_input)
ann_prediction = ann_model.predict_classes(user_input)

# Display the predictions
print("\nPredictions:")
print(f"KNN Prediction: {'Yes' if knn_prediction[0] else 'No'}")
print(f"Decision Tree Prediction: {'Yes' if dt_prediction[0] else 'No'}")
print(f"XGBoost Prediction: {'Yes' if xgb_prediction[0] else 'No'}")
print(f"ANN Prediction: {'Yes' if ann_prediction[0][0] else 'No'}")

# Calculate accuracy on the test set
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
ann_accuracy = accuracy_score(y_test, ann_model.predict_classes(X_test))

# Create a comparison table
table = tabulate([['KNN', knn_accuracy],
                 ['Decision Tree', dt_accuracy],
                 ['XGBoost', xgb_accuracy],
                 ['ANN', ann_accuracy]],
                headers=['Model', 'Accuracy'], tablefmt='pretty')

# Display the table
print("\nAccuracy on Test Set:")
print(table)
