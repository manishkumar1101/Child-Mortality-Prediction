# Child Mortality Prediction using Machine Learning Techniques

# Table of Contents
Introduction
Abstract
Problem Statement
Implementation
Algorithms Used
Proposed System
Results
Performance Analysis and Comparison
Conclusion
Setup and Usage

# Introduction
Artificial Intelligence (AI) focuses on enabling computers to perform tasks without explicit programming. Machine Learning (ML), a core part of AI, has numerous applications, including healthcare, where it can assist in predicting outcomes and improving treatments. This project applies ML techniques to predict child mortality, a significant global health concern, by analyzing complex health-related data.

# Abstract
Child mortality refers to the death of children under the age of 5. This research utilizes machine learning algorithms to classify and predict child mortality risks, helping to identify factors contributing to these fatalities. The study employs Naive Bayes, Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, and Support Vector Machine (SVM) algorithms. The dataset includes factors like demographics, health history, and environmental conditions, enabling a comprehensive prediction model.

# Problem Statement
Child mortality, particularly for children under five, remains a pressing issue worldwide. Traditional methods for identifying at-risk children are often slow and prone to errors, especially in resource-limited areas. This project aims to develop an ML model that can predict child mortality risks accurately, identifying high-risk children and contributing factors to enable timely interventions and reduce preventable deaths.

# Implementation
Data Collection and Integration
Data is gathered from healthcare facilities, public health databases, and community surveys, covering demographic details, medical history, nutritional status, environmental factors, and healthcare access.

# Feature Extraction
Key features like demographics, health history, and nutrition are refined, normalized, and cleaned. Missing values are handled to improve the model's predictive accuracy.

# Model Development
The dataset is split into training, validation, and test sets. Several ML algorithms are trained, and hyperparameters are optimized. The model's performance is evaluated using accuracy and precision metrics. After validation, the chosen model is deployed for real-time prediction with ongoing monitoring.

# Personalization of Interventions
By analyzing predictive data, healthcare providers can customize interventions based on demographics, health history, and environmental conditions, focusing resources on high-risk children.

# Evaluation and Validation
Performance is evaluated using metrics like accuracy, precision, recall, and F1-score. Cross-validation ensures that the model generalizes well, and ongoing monitoring is conducted to maintain accuracy.

# Algorithms Used
# Logistic Regression:
Estimates the probability of a child being at risk of mortality.
# Decision Trees:
Provides a clear and interpretable decision structure.
# Random Forest:
Uses ensemble learning to improve prediction accuracy.
# Support Vector Machines (SVM):
Finds an optimal hyperplane for classification.
# K-Nearest Neighbors (KNN): 
Classifies based on the majority class of nearest neighbors.
# Naive Bayes:
Applies Bayes' theorem, assuming feature independence.

# Proposed System
The proposed system primarily uses Logistic Regression due to its accuracy. Additionally, Support Vector Machines and data visualization methods are applied to provide insights into prediction accuracy, using line charts, pie charts, and bar graphs.

# Results
Initial setup includes loading libraries, dataset, and checking for null values.
Visualization through graphs helps understand data distribution.
Data is split into 80:20 for training and testing.
Algorithms like Logistic Regression, Random Forest, KNN, SVM, and Naive Bayes are applied. Logistic Regression provided the best accuracy, predicting child mortality with a high or low death ratio.
Performance Analysis and Comparison
# Algorithm	Accuracy (%)
Logistic Regression	68.8
Naive Bayes	64.4
SVM	63.4
Random Forest	66.8
K-Nearest Neighbors	46.6
# Conclusion
The project involved cleaning and preprocessing data, exploratory analysis, and model building. Logistic Regression demonstrated the highest accuracy in predicting child mortality. This model holds potential for real-world applications, assisting healthcare providers in identifying high-risk children and providing timely interventions.

# Setup and Usage
# Clone the Repository:
git clone https://github.com/yourusername/child-mortality-prediction.git
# Install Dependencies: Ensure you have Python and the necessary libraries:
pip install -r requirements.txt
# Run the Project:
python manage.py runserver
# Dataset:
Place the dataset file in the designated data/ folder before running the model.

# Results:
Results will display in the console, and visualizations of accuracy will appear in the output folder.

