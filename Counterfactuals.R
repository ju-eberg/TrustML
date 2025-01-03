library(dplyr)
library(tidyr)
library(ggplot2)


LendingAcc <- read.csv("C:/Users/jeroe/Uni_C/TrustML/LendingClub_2007_to_2018Q4.csv/accepted_2007_to_2018Q4.csv")
head(LendingAcc)

LendingRej <- read.csv("C:/Users/jeroe/Uni_C/TrustML/LendingClub_2007_to_2018Q4.csv/rejected_2007_to_2018Q4.csv")
head(LendingRej)


# Füge eine Spalte "Accepted" hinzu (1 für LendingAcc, 0 für LendingRej)
LendingAcc <- LendingAcc %>%
  mutate(Accepted = 1)

LendingRej <- LendingRej %>%
  mutate(Accepted = 0)

LendingAcc_relevant <- LendingAcc %>%
  select(
    loan_amnt, issue_d, title, 
    fico_range_low, fico_range_high,
    dti, zip_code, addr_state, emp_length, policy_code, Accepted
  )




# Convert LendingAcc issue_d (Dec-2015 format) to MM-YYYY format
LendingAcc_relevant$issue_d <- as.character(LendingAcc_relevant$issue_d)

# Create a vector for month abbreviations
month_abbr <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Function to replace month abbreviation with corresponding number
convert_month_to_number <- function(date) {
  month_name <- substr(date, 1, 3)  # Extract the 3-character month abbreviation
  month_number <- match(month_name, month_abbr)  # Find the corresponding month number
  year <- substr(date, 5, 8)  # Extract the year
  return(sprintf("%02d-%s", month_number, year))  # Return MM-YYYY format
}

# Apply the function to LendingAcc_relevant$issue_d
LendingAcc_relevant$issue_d <- sapply(LendingAcc_relevant$issue_d, convert_month_to_number)

# Convert LendingRej Application.Date (2007-05-26 format) to MM-YYYY format
LendingRej$Application.Date <- format(as.Date(LendingRej$Application.Date, format = "%Y-%m-%d"), "%m-%Y")



# Convert `dti` and `Debt.To.Income.Ratio` to numeric
LendingAcc_relevant$dti <- as.numeric(LendingAcc_relevant$dti)
LendingRej$Debt.To.Income.Ratio <- as.numeric(gsub("%", "", LendingRej$Debt.To.Income.Ratio))




# Remove spaces from `zip_code` and `addr_state`, and convert to lowercase
LendingAcc_relevant$zip_code <- gsub(" ", "", LendingAcc_relevant$zip_code)
LendingRej$Zip.Code <- gsub(" ", "", LendingRej$Zip.Code)

LendingAcc_relevant$addr_state <- tolower(LendingAcc_relevant$addr_state)
LendingRej$State <- tolower(LendingRej$State)

# Create a new variable `score` by calculating the rounded average of `fico_range_low` and `fico_range_high`
LendingAcc_relevant$score <- round((LendingAcc_relevant$fico_range_low + LendingAcc_relevant$fico_range_high) / 2)

# Rename columns in LendingRej to match those in LendingAcc_relevant
LendingRej <- LendingRej %>%
  rename(
    loan_amnt = Amount.Requested,
    issue_d = Application.Date,
    title = Loan.Title,
    score = Risk_Score,  # Create `score` as the rounded average of `fico_range_low` and `fico_range_high`
    dti = Debt.To.Income.Ratio,
    zip_code = Zip.Code,
    addr_state = State,
    emp_length = Employment.Length,
    policy_code = Policy.Code
  )

LendingRej$dti <- as.numeric(gsub("%", "", LendingRej$dti))

# Remove `fico_range_low`, `fico_range_high`, `title`, and `policy_code` from LendingAcc_relevant
LendingAcc_relevant <- LendingAcc_relevant %>%
  select(-fico_range_low, -fico_range_high, -title, -policy_code)

# Remove `fico_range_low`, `fico_range_high`, `title`, and `policy_code` from LendingRej
LendingRej <- LendingRej %>%
  select( -title, -policy_code)




# Combine LendingAcc_relevant and LendingRej
LendingClub <- bind_rows(LendingAcc_relevant, LendingRej)

# Remove rows with missing values
LendingClub <- na.omit(LendingClub)



# Train a logistic regression model using glm()
# We use Accepted as the target variable and the rest as predictors

modellr <- glm(Accepted ~ loan_amnt + dti +  emp_length,
             data = LendingClub, 
             family = binomial())

# Check the model summary
summary(modellr)



# Get predicted probabilities
pred_prob <- predict(modellr, type = "response")

# Convert probabilities to binary outcomes (threshold of 0.5)
predicted_class <- ifelse(pred_prob > 0.5, 1, 0)

# Check the predicted outcomes for the first few observations
head(predicted_class)



# Create confusion matrix to evaluate the model performance
table(Predicted = predicted_class, Actual = LendingClub$Accepted)

# ROC curve and AUC
library(pROC)
roc_curve <- roc(LendingClub$Accepted, pred_prob)
plot(roc_curve)
auc(roc_curve)
