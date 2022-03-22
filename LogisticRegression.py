# Import libraries
import pickle
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Create path names
READ_PATH = "data/heart_disease.csv"
PICKLE_PATH = "data/heart_disease_data.pkl"
SAVE_PATH = "data/logistic_regression.pkl"

# Load csv dataset, convert it to a pickle file, then load the pickle file onto a new DataFrame
df = pd.read_csv(READ_PATH)
df.to_pickle(PICKLE_PATH)
hd = pd.read_pickle(PICKLE_PATH)

# Rename the target column
hd = hd.rename(columns={"HeartDiseaseorAttack": "HeartDisease"})

# Separate features from target column
X = hd.drop("HeartDisease", axis=1)
y = hd["HeartDisease"]

# Under-sampled Logistic Regression model, then split the data into training and testing sets
under_sample = RandomUnderSampler(sampling_strategy="majority")
X_over, y_over = under_sample.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

hd_under = pd.concat([X_over, y_over.to_frame()], axis=1)  # Concatenate under-sampled model into single DataFrame

log_model_us = LogisticRegression(max_iter=500)  # Create LR model
log_model_us.fit(X_train, y_train)  # Fit/train model

# Save the final model
with open(SAVE_PATH, 'wb') as pickle_save:
    pickle.dump(log_model_us, pickle_save)
