# Titanic Survival Prediction 

## Project Goal
The goal of this project is to build a machine learning model that predicts whether a passenger survived the Titanic disaster, based on various features like age, gender, ticket class, and more. 

## Dataset
Description:
Passenger data including demographic, socio-economic, and travel-related features. The binary target variable is Survived.

Features include:

Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, etc.

## Approach 
The project was completed in the following stages:

- Exploratory Data Analysis (EDA) to discover patterns and insights.

- Preprocessing & Feature Engineering to clean, transform, and prepare the data.

- Model Training & Evaluation using various classification models.

- Hyperparameter Tuning with GridSearchCV and Stratified K-Fold cross-validation.

- Ensemble Modeling using a Soft Voting Classifier for improved performance.

- Evaluation on a stratified unseen sample to estimate generalization.

## EDA (Exploratory Data Analysis)

### Target Distribution:

38% of passengers survived. 

### Univariate Analysis:

Histograms and boxplots for Age, Fare.

Count plots for Sex, Pclass, Embarked.

### Bivariate Analysis & Insights:

Sex: Females had a much higher survival rate.

Pclass: 1st class passengers survived more.

Age: Younger passengers had better chances of survival.

Embarked: C was slightly more associated with survival than S or Q.

### Missing Values:

Age, Embarked, and Fare had missing values.

Cabin was dropped due to excessive missing data.

## Preprocessing
Preprocessing was implemented via a ColumnTransformer within a Pipeline to ensure consistency and avoid data leakage.

### Numerical Features:

Age: Filled using the median grouped by Sex and Pclass.

Fare: Filled using overall median.

Scaling via StandardScaler.

### Categorical Features:

Sex, Embarked, Pclass: Encoded with OneHotEncoder.

Embarked: Missing values filled using most frequent value.

## Feature Engineering:

FamilySize: SibSp + Parch + 1

IsAlone: 1 if FamilySize == 1, else 0

## Dropped Features:

Name, Cabin, Ticket, and PassengerId were removed.

## Modeling & Techniques Used
Models were trained in pipelines and tuned using GridSearchCV with StratifiedKFold (5 folds).

## Individual Models Trained:

| Model               | Best Parameters                        | Accuracy | AUC Score |
| ------------------- | -------------------------------------- | -------- | --------- |
| Logistic Regression | `C=1`, `solver='liblinear'`            | \~79%    | 0.78      |
| Random Forest       | `n_estimators=200`, `max_depth=10`     | \~82%    | 0.81      |
| SVC (RBF Kernel)    | `C=1`, `kernel='rbf'`                  | \~82%    | 0.80      |
| XGBoost             | `learning_rate=0.1`, `n_estimators=50` | \~82%    | 0.81      |

## Metrics Used
Accuracy

Precision, Recall, F1-score

ROC AUC Score

Precision-Recall AUC

Confusion Matrix

## Ensemble Modeling – Voting Classifier
A Soft Voting Classifier was used to combine the four trained models, improving generalization and stability.

## Ensemble Performance:
AUC Score: 0.90

Accuracy on Unseen Stratified Sample (100 rows): 84.81%


-> Soft voting allows the model to average predicted probabilities, giving better performance than any individual model.

## Techniques Summary

EDA :	Pandas, Matplotlib, Seaborn
Preprocessing :	Pipelines, ColumnTransformer, Imputation, OneHotEncoding, Scaling
Modeling :	Logistic Regression, Random Forest, SVC, XGBoost
Evaluation :	Accuracy, F1-score, ROC AUC, PR AUC, Confusion Matrix
Tuning :	GridSearchCV, StratifiedKFold
Ensemble :	VotingClassifier (Soft Voting)
Sampling :	StratifiedShuffleSplit 



