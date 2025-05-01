# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: functions_k2.R
# Date: 2025-04-17
# Note: This script creates functions needed for 
#       simulating DRE for k=2 version 02
#       In version 01 we had a true model as logistic-exponential
#       which is what is typically used by practitioners. 
#       We want to simulate something other than that this time.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

source("A_nn_tunning.R")
source("Y_nn_tunning.R")

# Generate a design matrix
# -------------------------------------
gen_X <- function(p=3, rho=0.6, mu, n) {
  Xnames <- paste0("X", 1:p) 
  Sigma <- outer(1:p, 1:p, function(i,j) rho^abs(i-j))
  X <- MASS::mvrnorm(n = n, mu = mu, Sigma = Sigma) |> 
    data.frame() |> 
    `colnames<-`(Xnames)
}


# Generate A (MVN link):
# -------------------------------------
gen_A <- function(X, beta) {
  mu <- as.matrix(cbind(1,X)) %*% as.matrix(beta) 
  Z <- MASS::mvrnorm(n = 1, mu = as.vector(mu), Sigma = diag(nrow = nrow(X)))
  A <- ifelse(Z > 0, 1, 0)
}



# Generate Y (MVN link)
# --------------------------------
#gamma <- dplyr::if_else(X[,1]>0, 0.7, 0.1) #<-with trt heterogeneity
gen_Y <- function(gamma = 0.8, X, A) {
  mu <- exp(1 + 0.1*X[,1] + 0.2*X[,2] + 0.3*X[,3] + gamma * A) 
  Y <- MASS::mvrnorm(n = 1, mu = as.vector(mu), Sigma = diag(nrow = nrow(X)))
}



# Estimating Y
# --------------------------------
outcome_model_mvn <- function(dat, pi_hat, ymod_formula, ptype = "response") {
  
  Y <- dat$Y
  A <- dat$A
  delta_1 <- as.numeric(A==1)
  delta_0 <- as.numeric(A==0)
  
  g_1 <- lm(as.formula(ymod_formula), data = dat[A==1,])
  ghat_1 <- predict(g_1, newdata = dat, type = "response")
  
  g_0 <- lm(as.formula(ymod_formula), data = dat[A==0,])
  ghat_0 <- predict(g_0, newdata = dat, type = "response")
  
  d <- get_diff(ghat_1, delta_1, ghat_0, delta_0, pi_hat, Y)
  
  print(paste0("  MVN estimated diff means = ", round(d$diff_means, 3)))
  o <- list("g_1" = g_1, "g_0" = g_0, "ptype" = ptype,
            "ghat_1" = ghat_1, "ghat_0" = ghat_0, 
            "muhat_1" = d$muhat_1, "muhat_0" = d$muhat_0)
  o
}


# One Sim function
# --------------------------------
one_sim <- function(n=n, p=3, Xmu, beta, gamma, ymod_formula_os, amod_formula_os,
                    hidunits_os, eps_os, penals_os) {
  
  X <- gen_X(n=n, p=p, rho=rho, mu=Xmu)
  A <- gen_A(X=X, beta=beta)
  Y <- gen_Y(X=X, A=A, gamma=gamma)
  dat <- cbind(Y,A,X) 
  
  true_est <- 
    mean(exp(as.matrix(cbind(1,X)) %*% as.matrix(beta) + gamma)) - 
    mean(exp(as.matrix(cbind(1,X)) %*% as.matrix(beta)))
  print(paste0("  True diff means = ", round(true_est, 3)))
  
  # Estimating propensity (A) model
  tic()
  H_true <- lm(as.formula(amod_formula_os), data=dat)
  pscores_true <- predict(H_true)
  
  H_logit <- glm(as.formula(amod_formula_os), family=binomial(link="logit"), data=dat)
  pscores_logit <- predict(H_logit, type = "response")
  
  H_nn <- A_model_nn(a_func=amod_formula_os, dat=dat, 
                     hidunits=hidunits_os, eps=eps_os, penals=penals_os) 
  pscores_nn <- predict(H_nn, new_data = dat %>% select(-A), type = "raw") |> as.vector()
  
  toc <- toc(quiet=TRUE)
  print(paste0("  A nn time:", round((toc$toc[[1]]-toc$tic[[1]])/60,2), " mins"))
  
  # Estimating the outcome (Y) model
  tic()
  fit_true <- outcome_model_mvn(dat, pi_hat=pscores_true, ymod_formula=ymod_formula_os)
  fit_expo <- outcome_model_expo(dat, pi_hat=pscores_logit, ymod_formula=ymod_formula_os)
  fit_nn <- outcome_model_nn(dat, pi_hat=pscores_nn, ymod_formula=ymod_formula_os,
                             hidunits=hidunits_os, eps=eps_os, penals=penals_os)
  toc <- toc(quiet=TRUE)
  print(paste0("  Y nn time:", round((toc$toc[[1]]-toc$tic[[1]])/60,2), " mins"))  
  
  # Packing results into a row
  naive_est <- mean(Y[A==1]) - mean(Y[A==0])
  true_model_est <- fit_true$muhat_1 - fit_true$muhat_0
  expo_model_est <- fit_expo$muhat_1 - fit_expo$muhat_0
  nn_model_est <- fit_nn$muhat_1 - fit_nn$muhat_0
  myrow <- tibble(
    "true_model" = true_est,
    "naive_est" = naive_est,
    "true_model_est" = true_model_est,
    "expo_model_est" = expo_model_est,
    "nn_model_est" = nn_model_est
  )
  myrow
}



















































