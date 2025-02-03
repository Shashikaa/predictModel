import pandas as pd

# Load the dataset
data = pd.read_csv('risk_data.csv')
print(data.head())  # Display the first few rows of the dataset

# Check the shape of the dataset
print("Dataset shape:", data.shape)

# Display basic statistics
print(data.describe(include='all'))

# Check for missing values
print("Missing values in each column:\n", data.isnull().sum())


# Drop rows with missing values (if any)
data = data.dropna()

# Convert categorical columns to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['Location', 'Activity', 'Item_Type'], drop_first=True)

# Define features and target variable
X = data.drop('Risk_Level', axis=1)  # Features
y = data['Risk_Level']               # Target variable

print("Features:\n", X.head())
print("Target:\n", y.head())

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

print("Model training complete!")

from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate accuracy and other metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import pickle

# Save the trained model to a file
with open('risk_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully!")
