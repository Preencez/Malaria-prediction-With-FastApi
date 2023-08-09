#!/usr/bin/env python
# coding: utf-8

# ## Malaria Prediction Using Machine Learning

# ### Introduction
# 
# This project focuses specifically on harnessing the potential of machine learning to predict the presence or absence of malaria in individuals. Malaria is a life-threatening vector-borne disease, often presents with a range of symptoms that can vary in severity and complexity. Leveraging a dataset containing relevant features such as demographic information, medical history, and clinical symptoms, advanced machine learning algorithms will be employed to create a predictive model. By learning intricate patterns and relationships within the data, this model aims to provide accurate and timely predictions, enabling healthcare practitioners to swiftly identify and initiate appropriate interventions for individuals at risk of malaria. Such an approach holds the potential to revolutionize malaria diagnosis, contributing to more efficient healthcare delivery and ultimately leading to improved patient outcomes.

# ### Hypothesis
# 
# Null Hypothesis (H0): There is no association between fever and sex.
# 
# 
# Alternative Hypothesis (H1): There is an association between fever and sex.

# ### Business Questions

# Question 1: Are symptoms gender specific 
# 
# Question 2: Treatment response and gender
# 
# Question 3: are there Risk factors and gender

# In[59]:


#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import os
from joblib import dump



# In[2]:


#loading the dataset
mal = pd.read_csv('reported_numbers.csv')


# #### checking all the information of the dataset (head,tail,info,shape,null values,describe,duplicated values, dtypes)

# In[3]:


mal.head(10)


# In[4]:


mal.tail(10)


# In[5]:


mal.info()


# In[6]:


mal.shape


# In[7]:


mal.dtypes


# In[8]:


mal.describe().transpose()


# In[9]:


mal.duplicated().sum()


# In[10]:


mal = mal.drop_duplicates()


# In[11]:


mal.isna().sum()


# In[12]:


mal.shape


# In[13]:


mal = mal.drop(columns=["age"])


# In[14]:


mal


# #### Explanation of each column
# 
# sex: This column represents the gender or sex of the individual. It indicates whether the individual is male or female.
# 
# fever: This column indicates whether the individual has a fever (high body temperature), with "yes" representing the presence of fever and "no" representing its absence.
# 
# cold: This column indicates whether the individual has cold symptoms, such as chills or shivering, with "yes" indicating the presence of cold symptoms and "no" indicating their absence.
# 
# rigor: Similar to "cold," this column represents the presence or absence of shivering or chills, often associated with fever. "Yes" indicates presence, and "no" indicates absence.
# 
# fatigue: This column indicates whether the individual experiences fatigue or extreme tiredness, with "yes" indicating its presence and "no" indicating its absence.
# 
# headace: This likely represents a typo and should be corrected to "headache." This column indicates the presence or absence of a headache, with "yes" indicating the presence of a headache and "no" indicating its absence.
# 
# bitter_tongue: This column indicates whether the individual experiences a bitter taste in the mouth, with "yes" indicating its presence and "no" indicating its absence.
# 
# vomitting: This column indicates whether the individual is vomiting, with "yes" indicating the presence of vomiting and "no" indicating its absence.
# 
# diarrhea: This column indicates whether the individual has diarrhea, with "yes" indicating its presence and "no" indicating its absence.
# 
# Convulsion: This column indicates whether the individual is experiencing convulsions (involuntary muscle contractions or shaking), with "yes" indicating their presence and "no" indicating their absence.
# 
# Anemia: This column indicates whether the individual has anemia (a condition characterized by low red blood cell count), with "yes" indicating its presence and "no" indicating its absence.
# 
# jundice: This column indicates whether the individual has jaundice (yellowing of the skin and eyes), with "yes" indicating its presence and "no" indicating its absence.
# 
# cocacola_urine: This column indicates whether the individual has "Coca-Cola" colored urine, which could be a sign of certain medical conditions. "Yes" indicates its presence, and "no" indicates its absence.
# 
# hypoglycemia: This column indicates whether the individual has hypoglycemia (low blood sugar), with "yes" indicating its presence and "no" indicating its absence.
# 
# prostraction: This likely represents a typo and should be corrected to "prostration." Prostration refers to extreme physical weakness or exhaustion. "Yes" indicates its presence, and "no" indicates its absence.
# 
# hyperpyrexia: This column indicates whether the individual has hyperpyrexia (very high fever), with "yes" indicating its presence and "no" indicating its absence.
# 
# severe_maleria: This column indicates whether the individual has severe malaria, with "yes" indicating its presence and "no" indicating its absence.

# ### Performing EDA

# #### Univariant Analysis
# 

# In[15]:


# looking at frequency of distribution
print("\nFrequency Distribution:")
for column in mal.columns:
    if column != 'sex':
        print(mal[column].value_counts())
        plt.figure()
        mal[column].value_counts().plot(kind='bar')
        plt.title(f"Frequency Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()


# In[16]:


# Proportion Analysis
print("\nProportion Analysis:")
for column in mal.columns:
    if column != 'sex':
        prop_by_sex = mal.groupby(['sex', column]).size() / mal.groupby(['sex']).size()
        print(prop_by_sex)
        prop_by_sex.unstack().plot(kind='bar', stacked=True)
        plt.title(f"Proportion Analysis of {column} by Gender")
        plt.xlabel("Gender")
        plt.ylabel("Proportion")
        plt.xticks(rotation=0)
        plt.legend(title=column)
        plt.show()


# ### Hypothesis testing
# 
# 
# Null Hypothesis (H0): There is no association between fever and sex.
# Alternative Hypothesis (H1): There is an association between fever and sex.

# In[17]:


# Chi-Square Test of Independence (Fever and Sex)
crosstab = pd.crosstab(mal['fever'], mal['sex'])
chi2, p, dof, expected = chi2_contingency(crosstab)
print("Chi-Square Test of Independence for Fever and Sex:")
print(f"Chi-square value: {chi2}")
print(f"P-value: {p}")
print("---")


# Based on the results above, where the p-value is approximately 0.702, we do not have enough evidence to reject the null hypothesis (H0). Therefore, we would accept the null hypothesis and conclude that there is no significant association between fever and sex in the given data. In other words, the data does not provide strong support for the presence of a relationship between these two variables

# ### Business Questions answers
# 

# In[18]:


# Question 1: Are symptoms gender specific
gender_symptoms_count = pd.crosstab(mal["sex"], mal["fever"])
# Print or further analyze the results
print("Gender-specific symptom patterns:")
print(gender_symptoms_count)


# This table indicates the count of individuals in each gender category (Female, Male) and their corresponding symptom patterns based on whether they have fever (No or Yes). For example, among females, 144 individuals do not have fever, while 402 females have fever. Similarly, among males, 114 individuals do not have fever, and 340 males have fever. This provides insights into how symptoms are distributed across different genders in relation to fever.

# In[19]:


# Question 2: Treatment response and gender
treatment_response = mal.groupby("sex")[["vomitting", "diarrhea", "hyperpyrexia"]].apply(lambda x: (x == "yes").sum())
print("\nTreatment response and gender:")
print(treatment_response)


# This table displays the number of individuals in each gender category (Female, Male) who have experienced particular symptoms, such as vomiting, diarrhea, and hyperpyrexia. For instance, among females, 51 individuals have experienced vomiting, 198 individuals have had diarrhea, and 78 individuals have had hyperpyrexia. Similarly, among males, 50 individuals have experienced vomiting, 147 individuals have had diarrhea, and 78 individuals have had hyperpyrexia. This data highlights how treatment responses vary between genders for these specific symptoms.

# In[20]:


# Question 3: Risk factors and gender
risk_factors_gender = pd.crosstab([mal["sex"], mal["bitter_tongue"], mal["cocacola_urine"], mal["hypoglycemia"]], mal["severe_maleria"])
print("\nRisk factors and gender:")
print(risk_factors_gender)


# This table presents the distribution of severe malaria occurrences based on the presence or absence of specific risk factors and gender. For instance, among females with bitter tongue and cocacola urine, 30 have severe malaria while 79 do not. Among males with hypoglycemia and bitter tongue, 35 have severe malaria while 62 do not. The table provides insights into how different risk factors might relate to the occurrence of severe malaria across genders.

# ### Feature processing and Engineering

# In[21]:


# Check data imbalance
class_counts = mal['sex'].value_counts()
print(class_counts)

# Visualize data imbalance
plt.figure(figsize=(8, 6))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('sex')
plt.ylabel('Count')
plt.title('Distribution of sex')
plt.show()


# In[22]:


total_samples = len(mal)

# Iterate through each symptom or condition column
for column in mal.columns[1:]:  # Exclude the 'sex' column
    value_counts = mal[column].value_counts()
    distribution = value_counts / total_samples * 100
    
    print(f"{column.capitalize()} Distribution:")
    for value, percentage in distribution.items():
        print(f"- {value.capitalize()}: {percentage:.2f}%")


# Yes, based on the distribution percentages for each symptom or condition, we can say that there is a significant class imbalance in several of the columns. Specifically, if the distribution of one class is much higher or much lower than the other, it indicates an imbalance.
# 
# Here are some columns where there is a noticeable class imbalance:
# 
# Vomitting: Yes (10.19%) vs. No (89.81%)
# Hypoglycemia: Yes (83.89%) vs. No (16.11%)
# Hyperpyrexia: No (84.20%) vs. Yes (15.80%)
# Severe_maleria: No (67.58%) vs. Yes (32.42%)
# These columns have a significant disparity between the occurrence of "Yes" and "No" values, suggesting an imbalance that could potentially affect the performance of our machine learning models trained on this data. 

# ### Feature Creation

# In[23]:


# Creating a new feature: Combination of Symptoms
mal['fever_and_vomitting'] = (mal['fever'] == 'yes') & (mal['vomitting'] == 'yes')

# Create a new feature: Severity Score (assuming symptom severity scores)
symptom_severity = {'yes': 1, 'no': 0}  # Assigning a binary score for simplicity
severity_columns = ['fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue',
                    'vomitting', 'diarrhea', 'Convulsion', 'Anemia', 'jundice',
                    'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia',
                    'severe_maleria']

mal['severity_score'] = mal[severity_columns].replace(symptom_severity).sum(axis=1)
# Display the updated dataset
print(mal.head())


# ### Data Balancing 

# In[24]:


# Define the columns with class imbalance
imbalanced_columns = ['vomitting', 'hypoglycemia', 'hyperpyrexia', 'severe_maleria']

# Initialize oversampling and undersampling instances
ros = RandomOverSampler(random_state=42)
rus = RandomUnderSampler(random_state=42)

# Split the data into training and testing sets, and balance each imbalanced column individually
for col in imbalanced_columns:
    # Extract features without the target column
    X = mal.drop([col, 'severe_maleria'], axis=1)
    y = mal[col]  # Extract the target label for the current column
    
    # Apply random oversampling to the minority class
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Apply random undersampling to the majority class
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)
    
    # Split the balanced data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)
    
#printing the shape of the splitted dataset
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[25]:


X_train.isna().sum()


# ### handle missing Data

# In[26]:


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Define your categorical column names
categorical_columns = ['sex', 'fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 'vomitting', 'diarrhea', 'Convulsion', 'Anemia', 'jundice', 'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia']

# Handle missing values in categorical columns using most frequent strategy and one-hot encoding
categorical_imputer = SimpleImputer(strategy='most_frequent')
preprocessor = ColumnTransformer(
    transformers=[('cat', categorical_imputer, categorical_columns)],
    remainder='passthrough'
)
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)


# 
# ### Encoding Categorical columns

# In[27]:


# Perform one-hot encoding on categorical columns
categorical_columns = ['sex', 'fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 'vomitting', 'diarrhea', 'Convulsion', 'Anemia', 'jundice', 'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia']
encoder = OneHotEncoder(drop='first', sparse=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])


# In[28]:


# Define the name of the numerical column
numerical_column = 'severity_score'

# Combine encoded categorical features with the numerical feature
X_train_final = np.hstack((X_train_encoded, X_train[[numerical_column]]))
X_test_final = np.hstack((X_test_encoded, X_test[[numerical_column]]))



# ### Model 1
# #### RandomForestClassiier

# In[29]:


# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_final, y_train)


# In[30]:


# Perform predictions and evaluate the model
y_pred = clf.predict(X_test_final)
y_pred_proba = clf.predict_proba(X_test_final)[:, 1]

# Convert 'yes' and 'no' labels to numeric values (1 and 0)
label_encoder = LabelEncoder()
y_test_numeric = label_encoder.fit_transform(y_test)
y_pred_numeric = label_encoder.transform(y_pred)

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
precision = precision_score(y_test_numeric, y_pred_numeric, pos_label=1)
recall = recall_score(y_test_numeric, y_pred_numeric, pos_label=1)
f1 = f1_score(y_test_numeric, y_pred_numeric, pos_label=1)
roc_auc = roc_auc_score(y_test_numeric, y_pred_proba)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC AUC:", roc_auc)


# In[31]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test_numeric, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[32]:


# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test_numeric, y_pred_proba)

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.show()


# In[33]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(y_test_numeric, y_pred_numeric)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()



# ### Support Vector Machine (SVM) Classifier

# In[36]:


# Create and train a Support Vector Machine classifier
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train_final, y_train)

# Perform predictions
y_pred_svm = svm_classifier.predict(X_test_final)


# In[38]:


# Convert 'yes' and 'no' labels in y_pred_svm to numeric values (1 and 0)
y_pred_svm_numeric = label_encoder.transform(y_pred_svm)

# Calculate and print evaluation metrics
accuracy_svm = accuracy_score(y_test_numeric, y_pred_svm_numeric)
precision_svm = precision_score(y_test_numeric, y_pred_svm_numeric, pos_label=1)
recall_svm = recall_score(y_test_numeric, y_pred_svm_numeric, pos_label=1)
f1_svm = f1_score(y_test_numeric, y_pred_svm_numeric, pos_label=1)
roc_auc_svm = roc_auc_score(y_test_numeric, y_pred_svm_numeric)

print("Support Vector Machine Classifier Metrics:")
print("Accuracy:", accuracy_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1-Score:", f1_svm)
print("ROC AUC:", roc_auc_svm)


# In[43]:


# Convert 'yes' and 'no' labels to numeric values (1 and 0)
label_encoder = LabelEncoder()
y_pred_svm_numeric = label_encoder.fit_transform(y_pred_svm)

# Compute ROC curve and ROC area
fpr_svm, tpr_svm, _ = roc_curve(y_test_numeric, y_pred_svm_numeric)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Plot ROC curve
plt.figure()
plt.plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Support Vector Machine Classifier')
plt.legend(loc="lower right")
plt.show()





# ###  Gradient Boosting classifier

# In[45]:


# Create and train a Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X_train_final, y_train)

# Perform predictions
y_pred_gb = gb_classifier.predict(X_test_final)


# In[46]:


# Convert 'yes' and 'no' labels to numeric values (1 and 0) for evaluation
y_pred_gb_numeric = label_encoder.transform(y_pred_gb)


# In[47]:


# Calculate and print evaluation metrics
accuracy_gb = accuracy_score(y_test_numeric, y_pred_gb_numeric)
precision_gb = precision_score(y_test_numeric, y_pred_gb_numeric, pos_label=1)
recall_gb = recall_score(y_test_numeric, y_pred_gb_numeric, pos_label=1)
f1_gb = f1_score(y_test_numeric, y_pred_gb_numeric, pos_label=1)
roc_auc_gb = roc_auc_score(y_test_numeric, y_pred_gb_numeric)

print("Gradient Boosting Classifier Metrics:")
print("Accuracy:", accuracy_gb)
print("Precision:", precision_gb)
print("Recall:", recall_gb)
print("F1-Score:", f1_gb)
print("ROC AUC:", roc_auc_gb)


# In[48]:


# Compute confusion matrix for Gradient Boosting classifier
cm_gb = confusion_matrix(y_test_numeric, y_pred_gb_numeric.round())

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt=".0f", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Gradient Boosting Classifier Confusion Matrix")
plt.show()


# In[49]:


# Compute ROC curve for Gradient Boosting classifier
fpr_gb, tpr_gb, _ = roc_curve(y_test_numeric, y_pred_gb_numeric)
roc_auc_gb = auc(fpr_gb, tpr_gb)

# Plot ROC curve for Gradient Boosting classifier
plt.figure()
plt.plot(fpr_gb, tpr_gb, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_gb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Gradient Boosting Classifier')
plt.legend(loc="lower right")
plt.show()


# ### Metrics comparison

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a DataFrame to store evaluation metrics for each model
data = {
    'Model': ['Random Forest', 'Support Vector Machine', 'Gradient Boosting'],
    'Accuracy': [accuracy, accuracy_svm, accuracy_gb],
    'Precision': [precision, precision_svm, precision_gb],
    'Recall': [recall, recall_svm, recall_gb],
    'F1-Score': [f1, f1_svm, f1_gb],
    'ROC AUC': [roc_auc, roc_auc_svm, roc_auc_gb]
}

metrics_df = pd.DataFrame(data)
print(metrics_df)


# In[56]:


# Define colors for each model
colors = ['red', 'green', 'black']

# Plot the F1 scores
plt.figure(figsize=(8, 6))
plt.bar(metrics_df['Model'], metrics_df['F1-Score'], color=colors)
plt.xlabel('Model')
plt.ylabel('F1-Score')
plt.title('F1-Score for Different Models')
plt.xticks(rotation=45)
plt.show()

# Plot the ROC AUC scores
plt.figure(figsize=(8, 6))
plt.bar(metrics_df['Model'], metrics_df['ROC AUC'], color=colors)
plt.xlabel('Model')
plt.ylabel('ROC AUC Score')
plt.title('ROC AUC Score for Different Models')
plt.xticks(rotation=45)
plt.show()


# ### Hyperparameter tunning of the best model
# RandomForestClassifier

# In[55]:


from sklearn.model_selection import GridSearchCV

# Create a Random Forest classifier
best_model = RandomForestClassifier(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy')

# Perform hyperparameter tuning
grid_search.fit(X_train_final, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)


# In[57]:


# Use the best model for prediction on test data
best_clf = grid_search.best_estimator_
y_pred_test = best_clf.predict(X_test_final)


# In[58]:


# Print the predicted labels
print("Predicted Labels:", y_pred_test)


# ### saving the best Model

# In[66]:


# Specify the desired destination directory
destination = "D:/Projects/Malaria prediction With FastApi/Project Directory/Ml components"

# Create the destination directory if it doesn't exist
os.makedirs(destination, exist_ok=True)

# Export the best Random Forest model
best_model_filepath = os.path.join(destination, "best_rf_model.joblib")
dump(best_clf, best_model_filepath)

# Export the categorical imputer
categorical_imputer_filepath = os.path.join(destination, "categorical_imputer.joblib")
dump(categorical_imputer, categorical_imputer_filepath)

# Export the label encoder
encoder_filepath = os.path.join(destination, "label_encoder.joblib")
dump(label_encoder, encoder_filepath)

# Print the paths to the exported components
print(f"Best Random Forest Model exported to: {best_model_filepath}")
print(f"Categorical Imputer exported to: {categorical_imputer_filepath}")
print(f"Label Encoder exported to: {encoder_filepath}")


# In[ ]:




