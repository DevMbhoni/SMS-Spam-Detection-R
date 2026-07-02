# SMS Spam Detection in R

A statistical classification project built in **R** using **RStudio**.

This project uses text preprocessing and machine learning models to classify SMS messages as either **spam** or **ham**.

---

## Project Overview

Spam detection is a binary classification problem where each SMS message is classified into one of two classes:

- **Ham**: legitimate/non-spam message
- **Spam**: unwanted promotional or fraudulent message

The goal of this project is to build and compare multiple machine learning models using TF-IDF text features.

---

## Dataset

The dataset used in this project is the **SMS Spam Collection** dataset from the **UCI Machine Learning Repository**.

Dataset source:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

According to the UCI Machine Learning Repository, the SMS Spam Collection is a public set of SMS labeled messages collected for mobile phone spam research. The dataset is commonly used for classification and clustering tasks.

In this project run, the loaded dataset contained:

| Class | Count |
|---|---:|
| Ham | 4,123 |
| Spam | 650 |
| Total | 4,773 |

---

## Technologies Used

- R
- RStudio
- tidyverse
- tidymodels
- tidytext
- textrecipes
- stopwords
- discrim
- naivebayes
- glmnet
- kernlab
- ranger
- xgboost
- vip

---

## Project Structure

```text
SpamDetectionR/
│
├── data/
│   └── SMSSpamCollection
│
├── scripts/
│   ├── 01_load_data.R
│   ├── 02_preprocessing.R
│   ├── 03_train_models.R
│   ├── 04_evaluate_models.R
│   └── 05_predict_sms.R
│
├── models/
│   ├── naive_bayes_model.rds
│   ├── logistic_regression_model.rds
│   ├── svm_model.rds
│   ├── random_forest_model.rds
│   └── xgboost_model.rds
│
├── outputs/
│   ├── sms_data_clean_raw.rds
│   ├── preprocessing_objects.rds
│   ├── model_metrics_results.csv
│   └── confusion_matrices/
│
└── README.md
```

---

## Machine Learning Pipeline

The project follows this workflow:

1. Load the SMS dataset
2. Convert labels into factor values
3. Check for missing values
4. Split the data into training and testing sets
5. Convert SMS text to lowercase
6. Tokenize SMS messages into words
7. Remove stopwords
8. Keep the top 1,000 tokens
9. Convert text into TF-IDF features
10. Train multiple classification models
11. Evaluate models using classification metrics
12. Save trained models
13. Predict new SMS messages

---

## Text Preprocessing

The raw SMS messages are converted into numerical features using a text preprocessing recipe.

```r
sms_recipe <- recipe(label ~ message, data = train_data) %>%
  step_mutate(message = stringr::str_to_lower(message)) %>%
  step_tokenize(message) %>%
  step_stopwords(message) %>%
  step_tokenfilter(message, max_tokens = 1000) %>%
  step_tfidf(message) %>%
  step_zv(all_predictors())
```

### TF-IDF

TF-IDF stands for **Term Frequency-Inverse Document Frequency**.

It gives higher importance to words that are useful for identifying a message, while reducing the effect of very common words that appear too often.

For example, words such as:

- free
- win
- prize
- claim
- urgent

may be more useful for identifying spam messages.

---

## Models Used

Five machine learning models were trained and compared.

---

### 1. Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' Theorem.

It calculates the probability that a message belongs to a class based on the words inside the message.

In spam detection, it estimates:

```text
P(spam | words in message)
```

Naive Bayes is commonly used for text classification because it is fast, simple, and effective with word-based features.

---

### 2. Logistic Regression

Logistic Regression is a classification model that predicts the probability of an input belonging to a class.

For this project, Logistic Regression predicts the probability that an SMS message is spam.

If the probability is high enough, the message is classified as spam. Otherwise, it is classified as ham.

Logistic Regression is useful because it is:

- Fast
- Interpretable
- Reliable as a baseline model
- Effective with TF-IDF features

---

### 3. Support Vector Machine

Support Vector Machine, or SVM, tries to find the best decision boundary between spam and ham messages.

The model attempts to separate the two classes with the widest possible margin.

SVM is strong for text classification because text data usually creates many high-dimensional TF-IDF features.

---

### 4. Random Forest

Random Forest is an ensemble model that builds many decision trees and combines their predictions.

Each tree makes a classification, and the forest chooses the final class based on the majority vote.

Random Forest reduces overfitting compared to a single decision tree and can perform well as a comparison model.

---

### 5. XGBoost

XGBoost is a boosting-based ensemble model.

Unlike Random Forest, which builds trees independently, XGBoost builds trees one after another.

Each new tree tries to correct the mistakes made by previous trees.

XGBoost is powerful but requires careful tuning to avoid overfitting.

---

## Model Results

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.975 | 0.949 | 0.862 | 0.903 | 0.989 |
| SVM | 0.959 | 0.810 | 0.915 | 0.859 | 0.975 |
| XGBoost | 0.964 | 0.936 | 0.792 | 0.858 | 0.968 |
| Logistic Regression | 0.960 | 0.934 | 0.762 | 0.839 | 0.973 |
| Naive Bayes | 0.864 | NA | 0.000 | NA | 0.500 |

---

## Best Model

The best model based on F1-score was:

```text
Random Forest
```

Random Forest achieved the strongest balance between precision and recall in this project run.

The final selected model was saved as:

```text
models/random_forest_model.rds
```

---

## Example Predictions

Two test messages were passed into the final model.

---

### Example 1

Input:

```text
Congratulations! You have won a free prize. Claim now.
```

Prediction:

```text
spam
```

Prediction probability:

| Ham | Spam |
|---:|---:|
| 0.125 | 0.875 |

---

### Example 2

Input:

```text
HI All,A reminder of the register below please, the link is below.
```

Prediction:

```text
ham
```

Prediction probability:

| Ham | Spam |
|---:|---:|
| 0.730 | 0.270 |

---

## How to Run the Project

Run the scripts in this order inside RStudio:

```text
scripts/01_load_data.R
scripts/02_preprocessing.R
scripts/03_train_models.R
scripts/04_evaluate_models.R
scripts/05_predict_sms.R
```

---

## Package Installation

Before running the project, install the required packages:

```r
install.packages(c(
  "tidyverse",
  "tidymodels",
  "tidytext",
  "textrecipes",
  "stopwords",
  "discrim",
  "naivebayes",
  "glmnet",
  "kernlab",
  "ranger",
  "xgboost",
  "vip"
))
```

## Conclusion

This project demonstrates how machine learning can be used for SMS spam detection using R.

The text messages were transformed into TF-IDF features and five classification models were trained and compared:

- Naive Bayes
- Logistic Regression
- Support Vector Machine
- Random Forest
- XGBoost

The best performing model in this run was **Random Forest**, achieving the highest F1-score among the tested models.

---

## Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Cross-validation
- Testing more token limits
- Comparing n-grams
- Improving Naive Bayes preprocessing
- Deploying the model as a Shiny app
- Adding a simple web interface for real-time SMS classification

---

## Author
**Mbhoni Shipalana**

Computer Science & Statistics Graduate  
Aspiring Software Engineer & Data Analyst

📧 Email: shipalanambhoniii@gmail.com  
💼 LinkedIn: https://www.linkedin.com/in/mbhoni-shipalana-83b9b826b

---
