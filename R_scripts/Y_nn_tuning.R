# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: nn_tunning.R
# Date: 2025-04-04
# Note: This script deploys a function for fitting a Single-layer Neural Net for 
#       an outcome (Y) model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(tidymodels)
library(future)

Y_model_nn <- function(dat, y_func, hidunits, eps, penals, cvs=6, verbose=FALSE) {
  
  
  
  
  # this splits data into training and testing
  dat_split <- initial_split(dat, prop=0.75)
  dat_train <- training(dat_split)
  dat_test <- testing(dat_split)
  
  dat_train <- dat
  
  # create a recipe with model and data steps
  Y_recip <- 
    recipe(as.formula(y_func), data = dat_train) %>% 
    step_normalize(all_predictors())
  
  Y_model_nn <- 
    mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) |> #epochs = iterations
    set_engine("nnet", trace = 0) %>% #trace prevents extra logging
    set_mode("regression")
  
  nn_wflow <- 
    workflow() %>%
    add_model(Y_model_nn) %>%
    add_recipe(Y_recip)
  
  # select hyperparameter ranges
  penalty_range <- penalty() %>% range_set(c(log10(min(penals)), log10(max(penals))))
  nn_param <- 
    extract_parameter_set_dials(Y_model_nn) %>%
    update(hidden_units = hidden_units(hidunits), 
           epochs = epochs(eps), 
           penalty = penalty(penals)) 
  
  # split training set into folds for cross validation
  folds <- vfold_cv(dat_train, v=cvs)
  
  # select tune metric
  tune_metric <- metric_set(rmse)
  
  # tune model
  plan(multisession, workers = 6) #<-turn on parallel processing
  nn_tune <- 
    nn_wflow %>%
    tune_grid(folds, 
              grid = nn_param %>% grid_latin_hypercube(size = 500),
              param_info = nn_param,
              metrics = tune_metric,
              control = control_grid(parallel_over = "resamples", allow_par = TRUE))
  plan(sequential) #<-restore sequential processing
  if(verbose==TRUE){
    print(show_best(nn_tune, metric = "rmse") %>% select(-.config, -.estimator))
  }

  # fit final model with best parameter set
  
  final_nn_wflow <- 
    nn_wflow %>%
    finalize_workflow(select_best(nn_tune, metric = "rmse") %>% select(-.config))
  
  final_nn_fit <-
    final_nn_wflow %>%
    fit(dat_train)
  
  #Yhat <- predict(final_nn_fit, new_data = dat_train %>% select(-Y), type = "raw") 
  #o <- list(Yhat = Yhat, final_nn_fit = final_nn_fit)
}



