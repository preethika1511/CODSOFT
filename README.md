# CODSOFT
# Titanic Survival Prediction

This project aims to predict the survival of passengers on the Titanic using machine learning techniques. The dataset used for this project contains information about the passengers, including their demographics and ticket details.

## Overview
The goal of this project is to build a predictive model that determines whether a passenger survived the Titanic disaster based on various features such as age, sex, passenger class, and more. This project demonstrates data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning model building.

## Dataset
The dataset used in this project is the Titanic dataset, available on [Kaggle](https://www.kaggle.com/c/titanic/data). It includes the following columns:

- **PassengerId**: Unique ID for each passenger
- **Survived**: Survival status (0 = No, 1 = Yes)
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Name**: Passenger's name
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Fare paid
- **Cabin**: Cabin number (if available)
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Project Workflow

### 1. Data Exploration and Cleaning
- Load the dataset and inspect its structure.
- Handle missing values using imputation techniques.
- Convert categorical data into numerical form where necessary.
- Remove or transform irrelevant features.

### 2. Exploratory Data Analysis (EDA)
- Use visualizations to explore relationships between features.
- Generate summary statistics to understand data distribution.
- Identify trends that may impact survival probability.

### 3. Feature Engineering
- Create new features to enhance model performance.
- Perform encoding for categorical variables.
- Scale numerical features where necessary.

### 4. Model Building
- Train multiple machine learning models, including:
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - Support Vector Machines (SVM)
- Perform hyperparameter tuning to optimize model performance.
- Evaluate models using accuracy, precision, recall, and F1-score.

## Tools and Technologies Used
- **Python**: Primary programming language
- **Jupyter Notebook**: Interactive coding environment
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Seaborn & Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning model development

## Results and Conclusion
The Titanic Survival Prediction project successfully demonstrates how machine learning can be applied to real-world datasets. Through data preprocessing, feature engineering, and model training, an accurate predictive model was developed. The project highlights:

- The impact of different features (such as gender and class) on survival rates.
- The importance of handling missing data and choosing relevant features.
- The effectiveness of different machine learning models in classification problems.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/preethika1511/CODSOFT.git
   ```
2. Navigate to the project folder and open the Jupyter Notebook.
3. Install required dependencies using:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
4. Run the notebook to see the analysis and model training process.

## References
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---
Feel free to contribute to this project or provide feedback! 🚀
