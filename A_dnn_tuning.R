# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: A_dnn.R
# Date: 2025-05-17
# Note: This script deploys a function for fitting a Deep Neural Net for 
#       a propensity score (A) model 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(tidymodels)
library(future)
library(brulee)

A_model_dnn <- function(dat, a_func, p, eps, penals, cvs=6, verbose=FALSE) {
  
  hunits_1 <- p
  hunits_2 <- c(0, p)
  penalty_range <- penalty() %>% range_set(c(log10(min(penals)), log10(max(penals))))
  
  # dependent variable has to be factor for classification
  dat$A <- factor(dat$A)
  
  # this splits data into training and testing
  #dat_split <- initial_split(dat, prop=0.75)
  #dat_train <- training(dat_split)
  #dat_test <- testing(dat_split)
  
  dat_train <- dat
  
  # create a recipe with model and data steps
  A_recip <- 
    recipe(as.formula(a_func), data = dat_train) %>% 
    step_normalize(all_predictors())
  
  A_model_nn <- # we only tune epochs and penals
    mlp(hidden_units = hunits_1, penalty = tune(), epochs = tune()) %>%
    set_engine("brulee_two_layer", hidden_units_2 = tune(), stop_iter = 5) %>% 
    set_mode("classification")
  
  nn_wflow <- 
    workflow() %>%
    add_model(A_model_nn) %>%
    add_recipe(A_recip)
  
  # select hyperparameter ranges
  nn_param <- 
    extract_parameter_set_dials(A_model_nn) %>%
    update(#hidden_units = hidden_units(hunits_1),
           hidden_units_2 = hidden_units_2(hunits_2),
           penalty = penalty(penals),
           epochs = epochs(eps))
  
  # split training set into folds for cross validation
  folds <- vfold_cv(dat_train, v=cvs)
  
  # select tune metric
  tune_metric <- metric_set(roc_auc)
  
  # tune model
  plan(multisession, workers = 6) #<-turn on parallel processing
  nn_tune <- 
    nn_wflow %>%
    tune_grid(folds, 
              grid = nn_param %>% grid_latin_hypercube(size = 65, original = FALSE),
              param_info = nn_param,
              metrics = tune_metric,
              control = control_grid(parallel_over = "resamples", allow_par = TRUE))
  plan(sequential) #<-restore sequential processing
  if(verbose==TRUE){
    print(show_best(nn_tune, metric = "roc_auc") %>% select(-.estimator, -.config))
  }
  
  # fit final model with best parameter set
  
  final_nn_wflow <- 
    nn_wflow %>%
    finalize_workflow(select_best(nn_tune, metric = "roc_auc") %>% select(-.config))
  
  final_nn_fit <-
    final_nn_wflow %>%
    fit(dat_train)
  
  # pscores_nn <- predict(final_nn_fit, new_data = dat_train %>% select(-A), type = "raw") 
  # A <- dat[1:1000,]$A
  # summary(pscores_nn)
  # res <- data.frame(A=A, Ahat = round(pscores_nn))
  # count(res, A, Ahat)
  # pscores_nn
}
