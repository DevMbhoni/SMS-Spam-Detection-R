library(tidyverse)

# Load the SMS Spam Collection dataset
sms_data <- read_delim(
  file = "data/SMSSpamCollection",
  delim = "\t",
  col_names = c("label", "message"),
  show_col_types = FALSE
)

# Convert label into factor
sms_data <- sms_data %>%
  mutate(
    label = factor(label, levels = c("ham", "spam")),
    message = as.character(message)
  )

# Inspect data
print(glimpse(sms_data))
print(table(sms_data$label))

# Check missing values
print(colSums(is.na(sms_data)))

# Save clean raw data
saveRDS(sms_data, "outputs/sms_data_clean_raw.rds")

cat("Data loaded successfully.\n")
