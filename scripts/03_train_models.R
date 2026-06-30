library(tidyverse)
library(tidymodels)
library(textrecipes)
library(discrim)
library(naivebayes)
library(glmnet)
library(kernlab)
library(ranger)
library(xgboost)

set.seed(123)

# Load preprocessing objects
objects <- readRDS("outputs/preprocessing_objects.rds")

train_data <- objects$train_data
sms_recipe <- objects$recipe

# -----------------------------
# 1. Naive Bayes
# -----------------------------
nb_model <- naive_Bayes(
  mode = "classification",
  smoothness = 1,
  Laplace = 1
) %>%
  set_engine("naivebayes")

nb_workflow <- workflow() %>%
  add_recipe(sms_recipe) %>%
  add_model(nb_model)

cat("Training Naive Bayes...\n")
nb_fit <- fit(nb_workflow, data = train_data)

saveRDS(nb_fit, "models/naive_bayes_model.rds")


# -----------------------------
# 2. Logistic Regression
# -----------------------------
logistic_model <- logistic_reg(
  mode = "classification",
  penalty = 0.01,
  mixture = 1
) %>%
  set_engine("glmnet")

logistic_workflow <- workflow() %>%
  add_recipe(sms_recipe) %>%
  add_model(logistic_model)

cat("Training Logistic Regression...\n")
logistic_fit <- fit(logistic_workflow, data = train_data)

saveRDS(logistic_fit, "models/logistic_regression_model.rds")


# -----------------------------
# 3. Linear SVM
# -----------------------------
svm_model <- svm_linear(
  mode = "classification",
  cost = 1
) %>%
  set_engine("kernlab")

svm_workflow <- workflow() %>%
  add_recipe(sms_recipe) %>%
  add_model(svm_model)

cat("Training SVM...\n")
svm_fit <- fit(svm_workflow, data = train_data)

saveRDS(svm_fit, "models/svm_model.rds")


# -----------------------------
# 4. Random Forest
# -----------------------------
rf_model <- rand_forest(
  mode = "classification",
  trees = 500,
  mtry = 50,
  min_n = 5
) %>%
  set_engine("ranger", importance = "impurity")

rf_workflow <- workflow() %>%
  add_recipe(sms_recipe) %>%
  add_model(rf_model)

cat("Training Random Forest...\n")
rf_fit <- fit(rf_workflow, data = train_data)

saveRDS(rf_fit, "models/random_forest_model.rds")


# -----------------------------
# 5. XGBoost
# -----------------------------
xgb_model <- boost_tree(
  mode = "classification",
  trees = 500,
  tree_depth = 6,
  learn_rate = 0.05,
  loss_reduction = 0,
  sample_size = 0.80,
  mtry = 100,
  min_n = 5
) %>%
  set_engine("xgboost")

xgb_workflow <- workflow() %>%
  add_recipe(sms_recipe) %>%
  add_model(xgb_model)

cat("Training XGBoost...\n")
xgb_fit <- fit(xgb_workflow, data = train_data)

saveRDS(xgb_fit, "models/xgboost_model.rds")


cat("All models trained and saved successfully.\n")