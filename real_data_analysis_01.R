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
set.seed(1810)
source("functions_k2_01.R")


# nn parameter space for tuning
hidunits = c(5,20)
eps = c(50,150)
penals = c(0.001,0.01)


# get data
df <- read_csv("real_data/recoded_ASTR_t1.csv")
Y <- df$gh
X <- cbind(df$gender, df$age, df$partial_or_total_removal) |> 
     `colnames<-`(c("X1","X2","X3"))
A <- df$chemo
dat <- as.data.frame(cbind(Y,A,X)) |> na.omit()


# grab a hold out for OTR
# random_row <- round(runif(1, nrow(df)))
# dat_new <- dat[random_row,] |> dplyr::select(-A) |> `rownames<-`(NULL)
Y <- dat$Y
A <- dat$A


# specify models
pmodel <- "A~X1+X2+X3"
omodel <- "Y~X1+X2+X3"


# estimate A (propensity model)
H_logit <- glm(as.formula(pmodel), family=binomial(link="logit"), data=dat)
pscores_logit <- predict(H_logit, type = "response")
H_nn <- A_model_nn(a_func=pmodel, dat=dat, 
                   hidunits=hidunits, eps=eps, penals=penals, verbose=FALSE) 
pscores_nn <- predict(H_nn, new_data = dat %>% select(-A), type = "raw") |> as.vector()


# estimate Y (outcome model)
dat$Y[dat$Y<=0] <- 0.00001 #FIX THIS
delta_1 <- as.numeric(A==1)
delta_0 <- as.numeric(A==0)

  ### TURN TO OLS
  ### fit_expo <- estimate_Y_expo(dat, pi_hat=pscores_logit, ymod_formula=omodel)
g_1 <- glm(as.formula(omodel), family = gaussian(link="log"), data = dat[A==1,])
g_0 <- glm(as.formula(omodel), family = gaussian(link="log"), data = dat[A==0,])
muhat_1 <- mean(ghat_1 + (delta_1*(Y - ghat_1)/(pscores_logit))/(mean(delta_1/pscores_logit)), na.rm = TRUE)
muhat_0 <- mean(ghat_0 + (delta_0*(Y - ghat_0)/(1-pscores_logit))/(mean(delta_0/(1-pscores_logit))), na.rm = TRUE)
diff_means_exp <- muhat_1 - muhat_0

fit_nn <- estimate_Y_nn(dat, pi_hat=pscores_nn, ymod_formula=omodel,
                        hidunits=hidunits, eps=eps, penals=penals, 
                        verbose=FALSE)
muhat_1 <- mean(fit_nn$ghat_1 + (delta_1*(Y - fit_nn$ghat_1)/(pscores_nn))/(mean(delta_1/pscores_nn)), na.rm = TRUE)
muhat_0 <- mean(fit_nn$ghat_0 + (delta_0*(Y - fit_nn$ghat_0)/(1-pscores_nn))/(mean(delta_0/(1-pscores_nn))), na.rm = TRUE)
diff_means_nn <- muhat_1 - muhat_0

# compute OTR Vn
get_Vn <- function(g_1, g_0, X_new, from_model = "nn") {
  
  if(from_model=="nn"){
    V_1 <- predict(g_1, new_data = X_new, type = "raw")
    V_0 <- predict(g_0, new_data = X_new, type = "raw")
  } else {
    V_1 <- predict(g_1, newdata = X_new, type = "response")
    V_0 <- predict(g_0, newdata = X_new, type = "response")    
  }
  r <- 
    tibble(X_new) |> 
    mutate(V_1 = V_1, 
           V_0 = V_0,
           Optimal_A = if_else(max(V_1, V_0) == V_1, 1, 0))
  
}

Vn_df_expo <- get_Vn(g_1 = fit_expo$g_1, 
                     g_0 = fit_expo$g_0, 
                     X_new = tibble(X1=1, X2=75, X3=0),
                     from_model = "expo") 

Vn_df_nn <- get_Vn(g_1 = fit_nn$g_1, 
                   g_0 = fit_nn$g_0, 
                   X_new = tibble(X1=1, X2=75, X3=1),
                   from_model = "nn") 
