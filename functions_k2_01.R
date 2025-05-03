# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: functions_k2.R
# Date: 2025-04-17
# Note: This script creates functions needed for 
#       simulating DRE for k=2 version 01
#       We simulate A with logistic link
#       and Y with exponential link
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(tictoc)
source("A_nn_tunning.R")
source("Y_nn_tunning.R")

# --------------------------
# Generate X (design matrix)
# --------------------------
gen_X <- function(p=3, rho=0.6, mu, n) {
  Xnames <- paste0("X", 1:p) 
  Sigma <- outer(1:p, 1:p, function(i,j) rho^abs(i-j))
  X <- MASS::mvrnorm(n = n, mu = mu, Sigma = Sigma) |> 
    data.frame() |> 
    `colnames<-`(Xnames)
}


# -----------
# Generate A 
# -----------

gen_A <- function(X, beta_A, flavor_A) {
  
  # logit link
  if(flavor_A == "logit") {
    expit <- function(x, b) {
      1/(1 + exp(-1*(x%*%b)))
    }
    A <- rbinom(n, 1, expit(x=as.matrix(cbind(1,X)), b=beta_A)) 
  }
  
  # gamma cdf link
  if(flavor_A == "beta") {
    xb <-(as.matrix(cbind(1,X))%*%beta_A)
    A <- rbinom(n, 1, pgamma(abs(xb), shape = 1.35, scale = 2)) 
  }
  A
}


# ------------
# Generate Y 
# ------------
#gamma <- dplyr::if_else(X[,1]>0, 0.7, 0.1) #<-with trt heterogeneity
gen_Y <- function(gamma, X, A, beta_Y, flavor_Y) {
  
  # exponential form
  if(flavor_Y == "expo"){
    lambda <- exp(as.matrix(cbind(1,X))%*%beta_Y + gamma * A) 
    Y <- rexp(n, rate = 1/lambda) #+ rnorm(n, 0, 0.001)
  }
  
  # square
  if(flavor_Y == "square"){
    Y <- (as.matrix(cbind(1,X))%*%beta_Y + gamma * A)^2 + abs(rnorm(n, 0, 0.01))
  }
  Y
}



# -----------------------------
# Estimate difference in means
# -----------------------------
get_diff <- function(ghat_1, delta_1, ghat_0, delta_0, pi_hat, Y) {
  muhat_1 <- mean(ghat_1 + (delta_1*(Y - ghat_1)/(pi_hat))/(mean(delta_1/pi_hat)))
  muhat_0 <- mean(ghat_0 + (delta_0*(Y - ghat_0)/(1-pi_hat))/(mean(delta_0/(1-pi_hat))))
  diff_means <- muhat_1 - muhat_0
  o <-list(diff_means = diff_means, muhat_1 = muhat_1, muhat_0 = muhat_0)
}

# --------------
# Estimating Y
# --------------
estimate_Y_nn <- function(dat, pi_hat, ymod_formula, verbose=FALSE,
                          hidunits=c(5,25), eps=c(50,200), penals=c(0.001,0.01)) {
  
  Y <- dat$Y
  A <- dat$A
  delta_1 <- as.numeric(A==1)
  delta_0 <- as.numeric(A==0)
  
  g_1 <- Y_model_nn(dat=dat[A==1, ], y_func = ymod_formula, 
                    hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
  ghat_1 <- predict(g_1, new_data = dat %>% select(-Y), type = "raw") |> as.vector()
  
  g_0 <- Y_model_nn(dat=dat[A==0, ], y_func = ymod_formula,
                    hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
  ghat_0 <- predict(g_0, new_data = dat %>% select(-Y), type = "raw") |> as.vector()
  
  d <- get_diff(ghat_1, delta_1, ghat_0, delta_0, pi_hat, Y)
  
  cat(paste0("\n  NN estimated diff means = ", round(d$diff_means, 3)))
  o <- list("g_1" = g_1, "g_0" = g_0, "ptype" = "raw",
            "ghat_1" = ghat_1, "ghat_0" = ghat_0, 
            "muhat_1" = d$muhat_1, "muhat_0" = d$muhat_0)
  o
  
}

estimate_Y_expo <- function(dat, pi_hat, ymod_formula) {
  
  Y <- dat$Y
  A <- dat$A
  delta_1 <- as.numeric(A==1)
  delta_0 <- as.numeric(A==0)
  
  g_1 <- glm(as.formula(ymod_formula), family = gaussian(link="log"), data = dat[A==1,])
  ghat_1 <- predict(g_1, newdata = dat, type = "response")
  
  g_0 <- glm(as.formula(ymod_formula), family = gaussian(link="log"), data = dat[A==0,])
  ghat_0 <- predict(g_0, newdata = dat, type = "response")
  
  
  #g_1 <- lm(log(Y) ~ X1 + X2 + X3, data = dat[A==1,])
  #ghat_1 <- predict(g_1, newdata = dat)
  #g_0 <- lm(log(Y) ~ X1 + X2 + X3, data = dat[A==0,])
  #ghat_0 <- predict(g_0, newdata = dat)
  
  d <- get_diff(ghat_1, delta_1, ghat_0, delta_0, pi_hat, Y)
  
  cat(paste0("\n  Exp estimated diff means = ", round(d$diff_means, 3)))
  o <- list("g_1" = g_1, "g_0" = g_0, "ptype" = "response",
            "ghat_1" = ghat_1, "ghat_0" = ghat_0, 
            "muhat_1" = d$muhat_1, "muhat_0" = d$muhat_0)
  o
}


# One Sim function
# --------------------------------
one_sim <- function(n=n, p=3, Xmu, beta_A, beta_Y, gamma, Y_fun, A_flavor, Y_flavor,
                    ymod_formula_os, amod_formula_os,
                    nn_hidunits, nn_eps, nn_penals, verbose = FALSE) {
  
  X <- gen_X(n=n, p=p, rho=rho, mu=Xmu)
  A <- gen_A(X=X, beta=beta_A, flavor_A=A_flavor)
  Y <- gen_Y(X=X, A=A, beta_Y=beta_Y, gamma=gamma, flavor_Y=Y_flavor)
  stopifnot(Y>0)
  print(paste0("  P(A)=",mean(A)))

  true_est <- 
    mean(Y_fun(as.matrix(cbind(1,X)) %*% as.matrix(beta_Y) + gamma)) - 
    mean(Y_fun(as.matrix(cbind(1,X)) %*% as.matrix(beta_Y)))
  cat(paste0("\n  True diff means = ", round(true_est, 3)))
  dat <- cbind(Y,A,X) 
  
  # Estimating A (propensity model)
  H_logit <- glm(as.formula(amod_formula_os), family=binomial(link="logit"), data=dat)
  pscores_logit <- predict(H_logit, type = "response")
  H_nn <- A_model_nn(a_func=amod_formula_os, dat=dat, 
                     hidunits=nn_hidunits, eps=nn_eps, penals=nn_penals, verbose=verbose) 
  pscores_nn <- predict(H_nn, new_data = dat %>% select(-A), type = "raw") |> as.vector()
  #toc <- toc(quiet=TRUE)
  #print(paste0("  A nn time:", round((toc$toc[[1]]-toc$tic[[1]])/60,2), " mins"))
  
  # Estimating Y (outcome model)
  fit_expo <- estimate_Y_expo(dat, pi_hat=pscores_logit, ymod_formula=ymod_formula_os)
  fit_nn <- estimate_Y_nn(dat, pi_hat=pscores_nn, ymod_formula=ymod_formula_os,
                          hidunits=nn_hidunits, eps=nn_eps, penals=nn_penals, 
                          verbose=verbose)
  #toc <- toc(quiet=TRUE)
  #print(paste0("  Y nn time:", round((toc$toc[[1]]-toc$tic[[1]])/60,2), " mins"))  
  
  
  # Packing results into a row
  naive_est <- mean(Y[A==1]) - mean(Y[A==0])
  expo_model_est <- fit_expo$muhat_1 - fit_expo$muhat_0
  nn_model_est <- fit_nn$muhat_1 - fit_nn$muhat_0
  myrow <- tibble(
    "prob_A" = mean(A),
    "true_diff" = true_est,
    "naive_est" = naive_est,
    "expo_model_est" = expo_model_est,
    "nn_model_est" = nn_model_est
  )
  myrow
}
