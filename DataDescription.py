# Import libraries
import LogisticRegression as LogReg
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# Import key components from LogisticRegression
MODEL_PATH = "data/logistic_regression.pkl"
lr_model = pickle.load(open(MODEL_PATH, 'rb'))

X_train = LogReg.X_train
X_test = LogReg.X_test
y_train = LogReg.y_train
y_test = LogReg.y_test

hd_original = LogReg.hd
hd_under = LogReg.hd_under

y_predictions = lr_model.predict(X_test)
y_predictions_prob = lr_model.predict_proba(X_test)[::, 1]

# List of age groups used for Age-to-Target plot legend
age_labels = ["18 - 24", "25 - 29", "30 - 34", "35 - 39", "40 - 44", "45 - 49", "50 - 54", "55 - 59", "60 - 64",
              "65 - 69", "70 - 74", "75 - 79", "80 and over"]

# Create DataFrame for age keys used in AgeToPhys graph
age_key = pd.DataFrame(columns=np.arange(1, 14, dtype=int))
age_key.loc[0] = age_labels  # Add age group fields to DataFrame
age_key = age_key.rename(index={0: 'Age'})

# Create classification report DataFrame to display in 'Descriptive' view
rpt = classification_report(y_test, y_predictions, output_dict=True)  # Evaluative classification report
classification_rpt = pd.DataFrame(rpt).transpose()  # Convert classification report to DataFrame for Streamlit Table

# Confusion Matrix
sns.set_theme(style="darkgrid")

fig, ax = plt.subplots(figsize=(4, 4))
ax = sns.heatmap(confusion_matrix(y_test, y_predictions),
                 annot=True,
                 annot_kws={'size': 15},
                 fmt='g',
                 cmap="viridis",
                 cbar=False)
plt.xlabel("Predictions", fontsize=14)
plt.ylabel("Truths", fontsize=14)
plt.title("Confusion Matrix", fontsize=16)
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
plt.close(fig)

# ROC Curve Plot
sns.set_theme(style="darkgrid", palette="Set2")

fpr, tpr, _ = roc_curve(y_test, y_predictions_prob)
auc = roc_auc_score(y_test, y_predictions_prob)
plt.plot(fpr, tpr, label=f'AUC: {round(auc, 3)}')
plt.title("ROC Curve", fontsize=16)
plt.legend()
plt.ylabel('True Positive Rate', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=14)
plt.savefig("images/roc_curve.png")
plt.close()

# Correlation Matrix
corr_matrix = hd_under.corr()
fig, ax = plt.subplots(figsize=(25, 16))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.7,
                 fmt=".2f",
                 annot_kws={"size": 17},
                 cmap="viridis")
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=18)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=18)
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=18)
plt.title("Correlation Matrix", fontsize=26)
plt.savefig("images/correlation_matrix.png")
corr_plot = ax.get_figure()
plt.close(fig)

# Age-to-Target Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax = sns.countplot(x="HeartDisease", hue="Age", data=hd_under, palette="viridis")
ax.set_title("Age Group Count by Target Class", fontsize=16)
ax.set_xticklabels(["No Disease", "Disease"])
ax.set_xlabel("Target Class", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
ax.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, labels=age_labels, title="Age Group")
plt.tight_layout()
plt.savefig("images/age_target_plot.png")
plt.close(fig)


# Function that creates graphs with specified age ranges.
def age_phys_age_range(min_age, max_age):
    # Age to Physical Health plot for negative target class
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=hd_under[(hd_under["HeartDisease"] == 0) &
                              (hd_under["Age"] >= int(min_age)) &
                              (hd_under["Age"] <= int(max_age))],
                x="Age", y="PhysHlth", ax=ax)
    ax.set_ylim(0, 20)
    plt.title("Relation of Age to Physical Health on NEGATIVE Target Class", fontsize=16)
    plt.ylabel('Monthly Days of Poor Phys Health', fontsize=14)
    plt.xlabel('Age', fontsize=14)
    plt.savefig("images/age_phys_plot0.png")
    plt.close(fig)

    # Age to Physical Health plot for positive target class
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=hd_under[(hd_under["HeartDisease"] == 1) &
                              (hd_under["Age"] >= min_age) &
                              (hd_under["Age"] <= max_age)],
                x="Age", y="PhysHlth", ax=ax)
    ax.set_ylim(0, 20)
    plt.title("Relation of Age to Physical Health on POSITIVE Target Class", fontsize=16)
    plt.ylabel('Monthly Days of Poor Phys Health', fontsize=14)
    plt.xlabel('Age', fontsize=14)
    plt.savefig("images/age_phys_plot1.png")
    plt.close(fig)


# Function that takes in user stated inputs and uses the trained Logistic Regression model to return the
# probability of being at risk for heart disease
def make_prediction(user_inputs):
    prediction_df = pd.DataFrame(user_inputs,
                                 columns=['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                                          'Diabetes', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
                                          'AnyHealthcare',
                                          'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age',
                                          'Education',
                                          'Income']
                                 )

    new_prediction = lr_model.predict_proba(prediction_df)[::, 1]
    probability = np.round(new_prediction[0] * 100, 2)
    return probability, prediction_df
