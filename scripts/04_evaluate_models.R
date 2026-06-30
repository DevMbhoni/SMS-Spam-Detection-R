library(tidyverse)
library(tidymodels)

# Load preprocessing objects
objects <- readRDS("outputs/preprocessing_objects.rds")
test_data <- objects$test_data

# Load trained models
models <- list(
  "Naive Bayes" = readRDS("models/naive_bayes_model.rds"),
  "Logistic Regression" = readRDS("models/logistic_regression_model.rds"),
  "SVM" = readRDS("models/svm_model.rds"),
  "Random Forest" = readRDS("models/random_forest_model.rds"),
  "XGBoost" = readRDS("models/xgboost_model.rds")
)

evaluate_model <- function(model_fit, model_name, test_data) {
  
  class_predictions <- predict(model_fit, test_data)
  prob_predictions  <- predict(model_fit, test_data, type = "prob")
  
  results <- test_data %>%
    select(label) %>%
    bind_cols(class_predictions) %>%
    bind_cols(prob_predictions)
  
  class_metric_results <- metric_set(
    accuracy,
    precision,
    recall,
    f_meas
  )(
    results,
    truth = label,
    estimate = .pred_class,
    event_level = "second"
  )
  
  roc_result <- roc_auc(
    results,
    truth = label,
    .pred_spam,
    event_level = "second"
  )
  
  final_metrics <- bind_rows(class_metric_results, roc_result) %>%
    mutate(model = model_name) %>%
    select(model, .metric, .estimate)
  
  cm <- conf_mat(
    results,
    truth = label,
    estimate = .pred_class
  )
  
  cm_table <- as_tibble(cm$table) %>%
    mutate(model = model_name)
  
  write_csv(
    cm_table,
    paste0("outputs/confusion_matrices/", str_replace_all(model_name, " ", "_"), "_confusion_matrix.csv")
  )
  
  return(final_metrics)
}

all_metrics <- map2_dfr(
  models,
  names(models),
  evaluate_model,
  test_data = test_data
)

all_metrics_wide <- all_metrics %>%
  pivot_wider(
    names_from = .metric,
    values_from = .estimate
  ) %>%
  arrange(desc(f_meas))

print(all_metrics_wide)

write_csv(all_metrics_wide, "outputs/model_metrics_results.csv")

best_model_name <- all_metrics_wide$model[1]

cat("\nBest model based on F1-score:", best_model_name, "\n")
cat("Evaluation completed successfully.\n")