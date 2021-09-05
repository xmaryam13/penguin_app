import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})

# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})

# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train,y_train)

def prediction(model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex):
  species = model.predict([['island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex']])
  species = species[0]
  if species == 0:
    return 'Adelie'
  elif species == 1:
    return 'Chinstrap'
  else:
    return 'Gentoo'

st.title("Penguin App")
b_length = st.slider("Bill length in mm", float(df['bill_length_mm'].min()),float(df['bill_length_mm'].max()))
b_depth = st.slider("Bill depth in mm", 0.0, 10.0)
f_length = st.slider("Flipper length in mm", 0.0, 10.0)
b_mass = st.slider("Body mass in g", 0.0, 10.0)

sex = st.selectbox('Gender',('Male','Female'))
if sex == 'Male':
  sex = 0
else:
  sex = 1
island = st.selectbox('Island',('Biscoe','Dream', 'Torgersen'))
if island == 'Biscoe':
  island = 0
elif island == 'Dream':
  island = 1
else:
  island = 2
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))
st.sidebar.button("Predict")
if classifier == 'Support Vector Machine': 
  species_type = prediction(svc_model, island, b_length, b_depth, f_length, b_mass, sex) 
  score = svc_score
elif classifier == 'Logistic Regression':
  species_type = prediction(log_reg,island,b_length,b_depth,f_length,b_mass,sex)
  score = log_reg
else:
  species_type = prediction(rf_clf,island,b_length,b_depth,f_length,b_mass,sex)
    