# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:24:13 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi'] = 300  # highres display


# Extract features
age = np.random.uniform(low=-10, high=10, size = 1000)


# Target variable
#f = 150-50*age-100*np.sin(age)+2*age**2+age**3+100*np.random.uniform(low=-1, high=1, size=len(age))
#f = 100-10*age+20*age**2+3*age**3+100*np.random.uniform(low=-1, high=1, size=len(age))
f = np.sin(age)+0.05*np.random.uniform(low=-1, high=1, size=len(age))

age = age[:,np.newaxis]
f = f[:,np.newaxis]

# Split the data into training/testing sets
age_train = age[0:200]
age_test = age[:]

# Split the targets into training/testing sets
f_age_train = f[0:200]
f_age_test = f[:]

# Create polynomial features and a linear regression model
degree = 11  # You can change the degree for different complexities
model = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression(fit_intercept=False))

# Train the model using the training sets
model.fit(age_train, f_age_train)

# Make predictions using the testing set
f_age_predict = model.predict(age_test)


sorted_indices = np.argsort(age_test.flatten())
age_test = (age_test.flatten())[sorted_indices]
f_age_test = (f_age_test.flatten())[sorted_indices]
f_age_predict = (f_age_predict.flatten())[sorted_indices]

# Access the linear regression model within the pipeline
linear_reg_model = model.named_steps['linearregression']

# Print the fitting parameters
print("Coefficients:\n", (linear_reg_model.coef_).flatten())
#print("Intercept:", linear_reg_model.intercept_)

# Plot outputs
plt.scatter(age_test, f_age_test, color='black', label='Actual data')
plt.plot(age_test, f_age_predict, color='blue', linewidth=2, label='Polynomial fit')
plt.xlabel('Age')
plt.ylabel('Disease Progression')
plt.title('Nonlinear Regression with Polynomial Features')
plt.legend()
plt.show()

print(f"\nError: {np.std(f_age_test-f_age_predict)}")