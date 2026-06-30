library(tidyverse)
library(tidymodels)

# Load your chosen model
# Change this if another model performs better after evaluation
best_model <- readRDS("models/random_forest_model.rds")

predict_sms <- function(message_text) {
  
  new_sms <- tibble(
    message = message_text
  )
  
  class_prediction <- predict(best_model, new_sms)
  probability_prediction <- predict(best_model, new_sms, type = "prob")
  
  result <- bind_cols(
    new_sms,
    class_prediction,
    probability_prediction
  )
  
  return(result)
}

example_1 <- predict_sms("Congratulations! You have won a free prize. Claim now.")
example_2 <- predict_sms("HI All,A reminder of the register below please, the link is below.")


print(example_1)
print(example_2)