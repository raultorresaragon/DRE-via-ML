## ----------------------------------------------------------------
## Q-Learning on the DynTxRegime bmiData toy dataset
## Two stages, binary treatments A1, A2
## Outcome to MINIMIZE: month12BMI (lower BMI = better)
## ----------------------------------------------------------------

library(DynTxRegime)

data(bmiData)
df <- as.data.frame(bmiData)
str(df)

to_numeric_recode <- function(x) {
  if (is.numeric(x)) return(x)
  x <- as.character(x)
  suppressWarnings(xn <- as.numeric(x))
  if (!any(is.na(xn))) return(xn)
  ## fallback: map two-level character/factor to -1/1
  lv <- sort(unique(x))
  if (length(lv) == 2) {
    return(ifelse(x == lv[1], 0, 1))
  }
  stop("Could not coerce variable to numeric 0/1")
}

df$A1 <- to_numeric_recode(df$A1)
df$A2 <- to_numeric_recode(df$A2)
df$gender <- to_numeric_recode(df$gender)
df$race   <- if (is.numeric(df$race)) df$race else as.numeric(as.factor(df$race))

set.seed(617)
#n <- nrow(df)             # should be 210
#n_train <- 190
#n_test  <- 20

#train_idx <- sample(seq_len(n), n_train)
#test_idx  <- setdiff(seq_len(n), train_idx)

#train <- df[train_idx, ]
#test  <- df[test_idx, ]

# Import train and test from the Python split to ensure
# we're working with the same split as DRE-ML
train <- read.csv('/Users/raul_torres_aragon/Library/CloudStorage/GoogleDrive-rdtaragon@gmail.com/My Drive/Dissertation/DRE-via-ML/python_scripts/RWD/train.csv')
train$A1 <- to_numeric_recode(train$A1)
train$A2 <- to_numeric_recode(train$A2)

test <- read.csv('/Users/raul_torres_aragon/Library/CloudStorage/GoogleDrive-rdtaragon@gmail.com/My Drive/Dissertation/DRE-via-ML/python_scripts/RWD/test.csv')
test$A1 <- to_numeric_recode(test$A1)
test$A2 <- to_numeric_recode(test$A2)

## ----------------------------------------------------------------
## STAGE 2 Q-learning model
## H2 = (gender, race, parentBMI, baselineBMI, A1, month4BMI): S2 covariates
## Outcome = month12BMI
## Include interactions with A2 so the rule depends on history
## ----------------------------------------------------------------

q2_model <- lm(
  month12BMI ~ (gender + race + parentBMI + baselineBMI + A1 + month4BMI) * A2,
  data = train
)

## Evaluate Q2 at A2 = 0 and A2 = 1 for every TRAINING patient
## (bmiData codes treatments as -1 / 1, not 0/1)
train_A2_neg <- train; train_A2_neg$A2 <- 0
train_A2_pos <- train; train_A2_pos$A2 <- 1

Q2_neg <- predict(q2_model, newdata = train_A2_neg)
Q2_pos <- predict(q2_model, newdata = train_A2_pos)

## We are MINIMIZING BMI -> optimal A2 is the one with LOWER predicted Q
d2_star_train <- ifelse(Q2_pos < Q2_neg, 1, 0)

## Pseudo-outcome: minimized Q2 for each training patient
Y2_tilde <- pmin(Q2_neg, Q2_pos)

## ----------------------------------------------------------------
## STAGE 1 Q-learning model
## H1 = (gender, race, parentBMI, baselineBMI)
## Response = pseudo-outcome Y2_tilde
## Include interactions with A1
## ----------------------------------------------------------------

train$Y2_tilde <- Y2_tilde

q1_model <- lm(
  Y2_tilde ~ (gender + race + parentBMI + baselineBMI) * A1,
  data = train
)

train_A1_neg <- train; train_A1_neg$A1 <- 0
train_A1_pos <- train; train_A1_pos$A1 <- 1

Q1_neg <- predict(q1_model, newdata = train_A1_neg)
Q1_pos <- predict(q1_model, newdata = train_A1_pos)

d1_star_train <- ifelse(Q1_pos < Q1_neg, 1, 0)

## ----------------------------------------------------------------
## APPLY LEARNED RULES TO THE TEST SET
## ----------------------------------------------------------------

## --- Stage 1 decision for test patients ---
test_A1_neg <- test; test_A1_neg$A1 <-  0
test_A1_pos <- test; test_A1_pos$A1 <-  1

Q1_neg_test <- predict(q1_model, newdata = test_A1_neg)
Q1_pos_test <- predict(q1_model, newdata = test_A1_pos)

d1_star_test <- ifelse(Q1_pos_test < Q1_neg_test, 1, 0)

## --- Stage 2 decision for test patients ---
## NOTE: we use the OBSERVED A1 and month4BMI from the test data,
## since these are determined by what actually happened to the patient
## up through stage 1 (Q-learning conditions on observed history,
## not on the recommended A1). This is standard practice when
## evaluating a learned regime on held-out trajectories that were
## not generated under the new regime.
test_A2_neg <- test; test_A2_neg$A2 <- 0
test_A2_pos <- test; test_A2_pos$A2 <- 1

Q2_neg_test <- predict(q2_model, newdata = test_A2_neg)
Q2_pos_test <- predict(q2_model, newdata = test_A2_pos)

d2_star_test <- ifelse(Q2_pos_test < Q2_neg_test, 1, 0)

## ----------------------------------------------------------------
## VALUE FUNCTION FOR THE TEST SET
##
## V(d*) = average of Q1(h1, d1_star) over test patients
## i.e., the predicted month12BMI under the OPTIMAL regime,
## using the stage-1 model (which already encodes the
## optimal stage-2 value via the pseudo-outcome).
## ----------------------------------------------------------------

Q1_star_test <- ifelse(d1_star_test == 1, Q1_pos_test, Q1_neg_test)

value_function <- mean(Q1_star_test)

## ----------------------------------------------------------------
## COMPARISON TO OBSERVED OUTCOME IN THE TEST SET
## ----------------------------------------------------------------

observed_mean_bmi <- mean(test$month12BMI)

cat("Estimated value function (predicted BMI under OTR):",
    round(value_function, 3), "\n")
cat("Observed mean month12BMI in test set:             ",
    round(observed_mean_bmi, 3), "\n")
cat("Difference (observed - OTR value):                 ",
    round(observed_mean_bmi - value_function, 3), "\n")

## ----------------------------------------------------------------
## SUMMARY TABLE OF RECOMMENDED VS OBSERVED TREATMENTS (test set)
## ----------------------------------------------------------------

results <- data.frame(
  #id          = test_idx,
  A1_observed = test$A1,
  A1_optimal  = d1_star_test,
  A2_observed = test$A2,
  A2_optimal  = d2_star_test,
  month12BMI_observed = test$month12BMI,
  Q_under_OTR = Q1_star_test
)

print(results)
print(paste0("V(d*)=", round(value_function,4)))
      