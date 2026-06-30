library(tidyverse)
library(tidymodels)
library(textrecipes)

set.seed(123)

# Load clean raw data
sms_data <- readRDS("outputs/sms_data_clean_raw.rds")

# Train/test split
sms_split <- initial_split(
  sms_data,
  prop = 0.80,
  strata = label
)

train_data <- training(sms_split)
test_data  <- testing(sms_split)

# Text preprocessing recipe
sms_recipe <- recipe(label ~ message, data = train_data) %>%
  step_mutate(message = stringr::str_to_lower(message)) %>%
  step_tokenize(message) %>%
  step_stopwords(message) %>%
  step_tokenfilter(message, max_tokens = 1000) %>%
  step_tfidf(message) %>%
  step_zv(all_predictors())

# Save objects for training
saveRDS(
  list(
    split = sms_split,
    train_data = train_data,
    test_data = test_data,
    recipe = sms_recipe
  ),
  "outputs/preprocessing_objects.rds"
)

cat("Preprocessing objects created successfully.\n")