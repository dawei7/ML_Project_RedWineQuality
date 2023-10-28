import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import pickle

# Read Data
data = pd.read_csv("./data/winequality-red.csv")

# If wine is rated between 3 and 6, it is not excellent. If wine is rated between 7 and 8, it is excellent.
data["excellent"] = data["quality"].apply(lambda x: 1 if x in [7, 8] else 0)
data.drop("quality", axis=1, inplace=True)

# Separate features and target
X = data.drop("excellent", axis=1)
y = data["excellent"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Initialize and train the Random Forest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train_normalized, y_train)

# Evaluate the model
predictions = random_forest_model.predict(X_test_normalized)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")

# Create a DataFrame with test set features, actual labels, and predicted labels
test_results = X_test.copy()
test_results['Actual_Label'] = y_test
test_results['Predicted_Label'] = pd.Series(predictions, index=X_test.index)

# Save the test results to a CSV file
test_results_filename = 'test_results.csv'
test_results.to_csv(test_results_filename, index=False)
print(f"Test results saved to {test_results_filename}")

# Serialize the model using pickle (Only if satisfied with the evaluation)
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(random_forest_model, file)

print(f"Model saved to {model_filename}")

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
