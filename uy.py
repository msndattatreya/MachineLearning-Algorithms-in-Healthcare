import pandas as pd
import random

# Generate random data for 100,000 records
data = {
    'Patient Id': list(range(1, 100001)),
    'Age': [random.randint(18, 90) for _ in range(100000)],
    'Systolic': [random.randint(100, 200) for _ in range(100000)],
    'Diastolic': [random.randint(60, 100) for _ in range(100000)],
    'Gender': [random.choice(['Male', 'Female']) for _ in range(100000)],
    'Cholesterol': [random.choice(['Normal', 'High']) for _ in range(100000)],
    'Smoking': [random.choice(['Yes', 'No']) for _ in range(100000)],
    'Family History': [random.choice(['Yes', 'No']) for _ in range(100000)],
    'Exercise': [random.choice(['Yes', 'No']) for _ in range(100000)],
    'Exang': [random.choice(['Yes', 'No']) for _ in range(100000)],
    'Heart Disease': [random.choice(['Yes', 'No']) for _ in range(100000)]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the dataset to an XLSX file
df.to_excel('Health_dataset.xlsx', index=False)

print("Data saved to Health_dataset.xlsx")
