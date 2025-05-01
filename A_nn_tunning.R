# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: nn_tunning.R
# Date: 2025-04-04
# Note: This script deploys a function for fitting a NNet for 
#       a propensity score model or an outcome model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(tidymodels)
library(future)
#library(doParallel)

A_model_nn <- function(dat, a_func, hidunits, eps, penals, cvs=8) {

  # dependent variable has to be factor for classification
  dat$A <- factor(dat$A)
  
  # this splits data into training and testing
  dat_split <- initial_split(dat, prop=0.75)
  dat_train <- training(dat_split)
  dat_test <- testing(dat_split)
  
  dat_train <- dat
  
  # create a recipe with model and data steps
  A_recip <- 
    recipe(as.formula(a_func), data = dat_train) %>% 
    step_normalize(all_predictors())
  
  A_model_nn <- 
    mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>% #epochs are iterations
    set_engine("nnet", trace = 0) %>% #trace prevents extra logging of the training process
    set_mode("classification")
  
  nn_wflow <- 
    workflow() %>%
    add_model(A_model_nn) %>%
    add_recipe(A_recip)
  
  # select hyperparameter ranges
  nn_param <- 
    extract_parameter_set_dials(A_model_nn) %>%
    update(hidden_units = hidden_units(hidunits), # hidden_units(c(5,25)),
           epochs = epochs(eps), # epochs(c(50,200)),
           penalty = penalty(penals)) # penalty(c(0.001,0.1)))
  
  # split training set into folds for cross validation
  folds <- vfold_cv(dat_train, v=cvs)
  
  # select tune metric
  tune_metric <- metric_set(roc_auc)
  
  # tune model
  #cl <- makeCluster(4)
  #registerDoParallel(cl)
  plan(multisession, workers = 4) #<-turn on parallel processing
  nn_tune <- 
    nn_wflow %>%
    tune_grid(folds, 
              grid = nn_param %>% grid_random(size = 500),
              param_info = nn_param,
              metrics = tune_metric,
              control = control_grid(parallel_over = "resamples", allow_par = TRUE))
  #stopCluster(cl)
  plan(sequential) #<-restore sequential processing
  (show_best(nn_tune, metric = "roc_auc") %>% select(-.estimator, -.config))
  
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



