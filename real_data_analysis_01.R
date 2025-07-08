# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: real_data_analysis_01.R
# Date: 2025-07-08
# Note: This script computes the OTR based on real data provided
#       by Dr. Ahn
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

library(tidyverse)
library(readr)
rm(list = ls())
source(here::here("functions_k2_01.R"))


df <- read_csv("real_data/recoded_ASTR_t1.csv")
dat <- 


# Estimating A (propensity model)
H_logit <- glm(as.formula(amod_formula_os), family=binomial(link="logit"), data=dat)
pscores_logit <- predict(H_logit, type = "response")
H_nn <- A_model_nn(a_func=amod_formula_os, dat=dat, 
                   hidunits=nn_hidunits, eps=nn_eps, penals=nn_penals, verbose=verbose) 
pscores_nn <- predict(H_nn, new_data = dat %>% select(-A), type = "raw") |> as.vector()


# Estimating Y (outcome model)
fit_expo <- estimate_Y_expo(dat, pi_hat=pscores_logit, ymod_formula=ymod_formula_os)
fit_nn <- estimate_Y_nn(dat, pi_hat=pscores_nn, ymod_formula=ymod_formula_os,
                        hidunits=nn_hidunits, eps=nn_eps, penals=nn_penals, 
                        verbose=verbose)


# Computing Vn
Vn_df <- get_Vn(g_1 = fit_nn$g_1, g_0 = fit_nn$g_0, X_new = X_new[1,]) 
