import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to read datasets (update paths as necessary)
def read_datasets():
    """ Reads users profile from csv files """
    genuine_users = pd.read_csv("data/users.csv")
    fake_users = pd.read_csv("data/fusers.csv")
    
    # Combine the datasets (assuming both datasets have the same structure)
    x = pd.concat([genuine_users, fake_users], ignore_index=True)
    
    # Create labels: 0 for genuine users, 1 for fake users
    y = np.concatenate([np.zeros(len(genuine_users)), np.ones(len(fake_users))])
    
    return x, y

# Function to extract features from the dataset
def extract_features(x):
    # Example feature engineering (this should be customized to your dataset)
    lang_list = list(enumerate(np.unique(x['lang'])))
    lang_dict = {name: i for i, name in lang_list}
    
    # Convert categorical language data to numerical data
    x['lang_code'] = x['lang'].map(lang_dict).astype(int)
    
    # Convert other features (update this based on your data structure)
    feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 
                              'favourites_count', 'listed_count', 'lang_code']
    
    # Handle missing values if any
    x = x[feature_columns_to_use].fillna(0)
    
    return x

# Read and preprocess the data
x, y = read_datasets()
x = extract_features(x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Predict on the test set
y_pred_prob = model.predict(X_test)  # Get predicted probabilities
y_pred = (y_pred_prob > 0.5).astype("int32")  # Convert probabilities to class labels

# Evaluate the model using accuracy
print('Accuracy:', accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Genuine', 'Fake'], yticklabels=['Genuine', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print("AUC Score:", roc_auc)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
