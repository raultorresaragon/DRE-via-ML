# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: double_robust.R
# Date: 2025-03-04
# Note: This script simulates data for a one-iteration
#       treatment regime with double robust estimators
#       using ML algorithms
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(123) 
rm(list = ls())
path <- paste0(getwd(),"/_Optimal Treatment Regime/Simulations")
n <- 1e3 + 100
source("A_nn_tunning.R")
source("Y_nn_tunning.R")

prop_model_formula <- "A~X1+X2+X3"
outcome_model_formula <- "Y~X1+X2+X3"


# ~~~~~~~~~~~~~~~~~ #
# 1.0 GENERATE DATA #
# ~~~~~~~~~~~~~~~~~ #

# Generate a design matrix
# -------------------------------------
gen_X <- function(p=3, rho=0.6, mu, n) {
  Xnames <- paste0("X", 1:p) 
  Sigma <- outer(1:p, 1:p, function(i,j) rho^abs(i-j))
  X <- MASS::mvrnorm(n = n, mu = mu, Sigma = Sigma) |> 
    data.frame() |> 
    `colnames<-`(Xnames)
}
p <- 3                        # number of covariates
rho <- 0.6                    # corr coeff for covariance of MVN X
mu <- c(-2, 0.1, 1)           # vector of means for covatiates
X <- gen_X(mu=mu, n=n)



# Generate A (logit link):
# -------------------------------------
gen_A <- function(X, beta) {
  expit <- function(x, b) {
    1/(1 + exp(-1*(x%*%b)))
  }
  A <- rbinom(n, 1, expit(x=as.matrix(cbind(1,X)), b=beta)) 
}
beta <- c(1, seq(from = 1*.1, to = p*.1, by = .1))
A <- gen_A(X=X, beta=beta)
print(paste0("P(A==1) = ", mean(A)))

# Generate Y (exponential link)
# --------------------------------
#gamma <- dplyr::if_else(X[,1]>0, 0.7, 0.1) #<-with trt heterogeneity
gen_Y <- function(gamma = 0.8, X, A) {
  lambda <- exp(1 + 0.1*X[,1] + 0.2*X[,2] + 0.3*X[,3] + gamma * A) 
  Y <- rexp(n, rate = 1/lambda) + rnorm(n, 0, 0.001)
  
}
Y <- gen_Y(X=X, A=A)
dat <- cbind(Y,A,X) 
glimpse(dat)


# split data into two sets
newdat <- dat[1:100, ]
dat <- dat[101:n, ]


# ~~~~~~~~~~~~~~~~~~~ #
# 2.0 ESTIMATE MODELS #
# ~~~~~~~~~~~~~~~~~~~ # 

# Estimating propensity (A) model
# ---------------------------
H_logit <- glm(as.formula(prop_model_formula), family=binomial(link="logit"), data=dat)
pscores_logit <- predict(H_logit, type = "response")
H_nn <- A_model_nn(a_func = prop_model_formula, dat = dat) 
pscores_nn <- predict(H_nn, new_data = dat %>% select(-A), type = "raw") |> as.vector()


# Estimating outcome (Y) model
# ------------------------
get_omodel <- function(dat, pi_hat, type, outcome_model_formula) {
  
  Y <- dat$A
  A <- dat$A
  delta_1 <- as.numeric(A==1)
  delta_0 <- as.numeric(A==0)
  
  if(type == "nn") {
    ptype = "raw"
    
    dat$A <- 1
    g_res1 <- Y_model_nn(dat=dat, y_func = outcome_model_formula)
    g_1 <- g_res1$final_nn_fit
    ghat_1 <- g_res1$Yhat
    
    dat$A <- 0
    g_res0 <- Y_model_nn(dat=dat, y_func = outcome_model_formula)
    g_0 <- g_res0$final_nn_fit
    ghat_0 <- g_res0$Yhat
    
  } else {
    ptype = "response"
    
    dat$A <- 1
    g_1 <- glm(as.formula(outcome_model_formula), family = gaussian(link="log"), data = dat)
    ghat_1 <- predict(g_1, type = "response")
    
    dat$A <- 0
    g_0 <- glm(as.formula(outcome_model_formula), family = gaussian(link="log"), data = dat)
    ghat_0 <- predict(g_0, type = "response")
  }
  
  #muhat_1 <- mean(ghat_1[A==1] + (Y[A==1] - ghat_1[A==1])/pi_hat[A==1])
  #muhat_0 <- mean(ghat_0[A==0] + (Y[A==0] - ghat_0[A==0])/(1-pi_hat[A==0]))
  muhat_1 <- mean(delta_1*ghat_1 + delta_1*(Y - ghat_1)/(pi_hat))
  muhat_0 <- mean(delta_0*ghat_0 + delta_0*(Y - ghat_0)/(1-pi_hat))
  (diff_means <- muhat_1 - muhat_0)
  
  print(paste0("estimated diff means = ", round(diff_means, 3)))
  o <- list("g_1" = g_1, "g_0" = g_0, "ptype" = ptype,
            "ghat_1" = ghat_1, "ghat_0" = ghat_0, 
            "muhat_1" = muhat_1, "muhat_0" = muhat_0)
  o
}

fit_expo <- get_omodel(dat, pi_hat = pscores_logit, type = "expo", outcome_model_formula)
fit_nn <- get_omodel(dat, pi_hat = pscores_nn, type = "nn", outcome_model_formula)

true_est <- mean(exp(as.matrix(cbind(1,X)) %*% as.matrix(theta) + gamma)) - 
            mean(exp(as.matrix(cbind(1,X)) %*% as.matrix(theta)))
naive_est <- mean(Y[A==1]) - mean(Y[A==0])
true_model_est <- fit_expo$muhat_1 - fit_expo$muhat_0
nn_model_est <- fit_nn$muhat_1 - fit_nn$muhat_0

model_results <- 
  tibble(model = character(), diff_estimate = numeric()) |>
  rbind(tibble(model = "true diff", diff_estimate = true_est)) |>
  rbind(tibble(model = "naive_est", diff_estimate = naive_est)) |>
  rbind(tibble(model = "true_model_est", diff_estimate = true_model_est)) |>
  rbind(tibble(model = "nn_model_est", diff_estimate = nn_model_est))
model_results

# ~~~~~~~~~~~~~~~~~~~~~~~~~ #
# 3.0 Obtain V(d) = E[G(Y)] #
# ~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Assume a larger Y is more desirable

# create a new dataset with n=25. Compute muhat_1 and muhat_0 for each i. Then take the 
# max(muhat_1, muhat_0) for each i. That is the optimal rule d*


get_Vn <- function(myfit, new_pi_hat, newdat) {
  n <- dim(newdat)[1]
  delta_1 <- as.numeric(newdat$A==1)
  delta_0 <- as.numeric(newdat$A==0)
  ghat_1 <- predict(myfit$g_1, newdat |> mutate(A==1), type = myfit$ptype)
  ghat_0 <- predict(myfit$g_0, newdat |> mutate(A==0), type = myfit$ptype)
  Vn_d1 <- mean(delta_1*ghat_1 + delta_1*(newdat$Y - ghat_1)/(new_pi_hat))
  Vn_d0 <- mean(delta_0*ghat_1 + delta_0*(newdat$Y - ghat_0)/(1-new_pi_hat))
  o <- list(Vn_d1 = Vn_d1, Vn_d0 = Vn_d0)
}

new_pi_hat_nn <- predict(H_nn, new_data = newdat %>% select(-A), type = "raw") |> as.vector()
new_pi_hat_logit <- predict(H_logit, new_data = newdat, type = "response")
Vn <- get_Vn(fit_expo, new_pi_hat_logit, newdat)
Vn$Vn_d1
Vn$Vn_d0
Vn <- get_Vn(fit_nn, new_pi_hat_nn, newdat)
Vn$Vn_d1
Vn$Vn_d0

save.image(file='Sim1.RData')








