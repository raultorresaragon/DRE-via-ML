# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: functions_k2.R
# Date: 2025-04-17
# Note: This script creates functions needed for 
#       simulating DRE for k=2 version 01
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


# Generate A (logit link):
# -------------------------------------
gen_A <- function(X, beta) {
  expit <- function(x, b) {
    1/(1 + exp(-1*(x%*%b)))
  }
  A <- rbinom(n, 1, expit(x=as.matrix(cbind(1,X)), b=beta)) 
}


# Generate Y (exponential link)
# --------------------------------
#gamma <- dplyr::if_else(X[,1]>0, 0.7, 0.1) #<-with trt heterogeneity
gen_Y <- function(gamma = 0.8, X, A) {
  lambda <- exp(1 + 0.1*X[,1] + 0.2*X[,2] + 0.3*X[,3] + gamma * A) 
  Y <- rexp(n, rate = 1/lambda) + rnorm(n, 0, 0.001)
  
}


# Estimating Y
# --------------------------------
get_diff <- function(ghat_1, delta_1, ghat_0, delta_0, pi_hat, Y) {
  muhat_1 <- mean(ghat_1 + (delta_1*(Y - ghat_1)/(pi_hat))/(mean(delta_1/pi_hat)))
  muhat_0 <- mean(ghat_0 + (delta_0*(Y - ghat_0)/(1-pi_hat))/(mean(delta_0/(1-pi_hat))))
  diff_means <- muhat_1 - muhat_0
  o <-list(diff_means = diff_means, muhat_1 = muhat_1, muhat_0 = muhat_0)
}

outcome_model_nn <- function(dat, pi_hat, ymod_formula, ptype = "raw", 
                             hidunits=c(5,25), eps=c(50,200), penals=c(0.001,0.01)) {
  
  Y <- dat$Y
  A <- dat$A
  delta_1 <- as.numeric(A==1)
  delta_0 <- as.numeric(A==0)
  
  g_1 <- Y_model_nn(dat=dat[A==1, ], y_func = ymod_formula, 
                    hidunits=hidunits, eps=eps, penals=penals)
  ghat_1 <- predict(g_1, new_data = dat %>% select(-Y), type = "raw") |> as.vector()
  
  g_0 <- Y_model_nn(dat=dat[A==0, ], y_func = ymod_formula,
                    hidunits=hidunits, eps=eps, penals=penals)
  ghat_0 <- predict(g_0, new_data = dat %>% select(-Y), type = "raw") |> as.vector()
  
  #muhat_1 <- mean(ghat_1 + delta_1*(Y - ghat_1)/(pi_hat))
  #muhat_0 <- mean(ghat_0 + delta_0*(Y - ghat_0)/(1-pi_hat))
  d <- get_diff(ghat_1, delta_1, ghat_0, delta_0, pi_hat, Y)
  
  print(paste0("  NN estimated diff means = ", round(d$diff_means, 3)))
  o <- list("g_1" = g_1, "g_0" = g_0, "ptype" = ptype,
            "ghat_1" = ghat_1, "ghat_0" = ghat_0, 
            "muhat_1" = d$muhat_1, "muhat_0" = d$muhat_0)
  o
  
}

outcome_model_expo <- function(dat, pi_hat, ymod_formula, ptype = "response") {
  
  Y <- dat$Y
  A <- dat$A
  delta_1 <- as.numeric(A==1)
  delta_0 <- as.numeric(A==0)
  
  g_1 <- glm(as.formula(ymod_formula), family = gaussian(link="log"), data = dat[A==1,])
  ghat_1 <- predict(g_1, newdata = dat, type = "response")
  
  g_0 <- glm(as.formula(ymod_formula), family = gaussian(link="log"), data = dat[A==0,])
  ghat_0 <- predict(g_0, newdata = dat, type = "response")
  
  d <- get_diff(ghat_1, delta_1, ghat_0, delta_0, pi_hat, Y)
  
  print(paste0("  Exp estimated diff means = ", round(d$diff_means, 3)))
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
  
  true_est <- 
    mean(exp(as.matrix(cbind(1,X)) %*% as.matrix(beta) + gamma)) - 
    mean(exp(as.matrix(cbind(1,X)) %*% as.matrix(beta)))
  print(paste0("  True diff means = ", round(true_est, 3)))
  
  # split data into two sets
  # fulldat <- cbind(Y,A,X) 
  # rand_i <- sample(1:n, ceiling(n*0.1))
  # newdat <- fulldat[rand_i, ]
  # dat <- fulldat[-rand_i, ]
  dat <- cbind(Y,A,X) 
  
  # Estimating propensity (A) model
  tic()
  H_logit <- glm(as.formula(amod_formula_os), family=binomial(link="logit"), data=dat)
  pscores_logit <- predict(H_logit, type = "response")
  H_nn <- A_model_nn(a_func=amod_formula_os, dat=dat, 
                     hidunits=hidunits_os, eps=eps_os, penals=penals_os) 
  pscores_nn <- predict(H_nn, new_data = dat %>% select(-A), type = "raw") |> as.vector()
  toc <- toc(quiet=TRUE)
  print(paste0("  A nn time:", round((toc$toc[[1]]-toc$tic[[1]])/60,2), " mins"))
  
  # Estimating the outcome (Y) model
  tic()
  fit_expo <- outcome_model_expo(dat, pi_hat=pscores_logit, ymod_formula=ymod_formula_os)
  fit_nn <- outcome_model_nn(dat, pi_hat=pscores_nn, ymod_formula=ymod_formula_os,
                             hidunits=hidunits_os, eps=eps_os, penals=penals_os)
  toc <- toc(quiet=TRUE)
  print(paste0("  Y nn time:", round((toc$toc[[1]]-toc$tic[[1]])/60,2), " mins"))  
  
  # Packing results into a row
  naive_est <- mean(Y[A==1]) - mean(Y[A==0])
  true_model_est <- fit_expo$muhat_1 - fit_expo$muhat_0
  nn_model_est <- fit_nn$muhat_1 - fit_nn$muhat_0
  myrow <- tibble(
    "true_model" = true_est,
    "naive_est" = naive_est,
    "true_model_est" = true_model_est,
    "nn_model_est" = nn_model_est
  )
  myrow
}
