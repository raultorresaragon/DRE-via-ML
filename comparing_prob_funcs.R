# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: comparing_prob_func.R
# Date: 2025-05-04
# Note: In order to model P(A=1), I need to construct a vector of probabilities
#       that I can then pass to rbinom() to map it to a vector of 0s and 1s.
#       The first logical option is the logistic function with range -inf to +inf
#       When modeling the true model,it's easy to use logistic regression, 
#       but I want to model P(A=1) with a different function to see how the 
#       typical choice (logistic) does against neural nets. 
#       This script compares multiple options I can use to model P(A=1)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rm(list = ls())
n <- 1000
p <- 3
rho   <- round(runif(1, 0.4, 0.6),1)
Xmu   <- round(runif(3, -1, 1),1) 
beta_A <- c(1, round(runif(3, -1, 1),1))
beta_Y <- c(1, round(runif(3, -1, 1),1))
gamma <- 0.6

gen_X <- function(p=3, rho=0.6, mu, n) {
  Xnames <- paste0("X", 1:p) 
  Sigma <- outer(1:p, 1:p, function(i,j) rho^abs(i-j))
  X <- MASS::mvrnorm(n = n, mu = mu, Sigma = Sigma) |> 
    data.frame() |> 
    `colnames<-`(Xnames)
}

X <- gen_X(p=3, rho=rho, mu=Xmu, n=n)
summary(X)

gen_probs <- function(X, beta_A) {
  xb <-(as.matrix(cbind(1,X))%*%beta_A) 
  probs_logit <- 1/(1 + exp(-1*(xb)))
  probs_norm <- pnorm(xb)
  probs_gompertz <- exp(-exp(xb))
  probs_arctan <- (atan(xb)/pi) + 0.5
  probs_tanh <- 0.5* (tanh(xb)+1)
  r <- list(probs_logit = probs_logit, 
            probs_norm = probs_norm, 
            probs_gompertz = probs_gompertz,
            probs_arctan = probs_arctan,
            probs_tanh = probs_tanh,
            xb = xb)  
}


p <- gen_probs(X, beta_A)

plot(y=p$probs_logit, x=(p$xb), col = "purple", ylim=c(0,1), xlim=c(-4,4))
points(y=p$probs_norm, x=(p$xb), col = "blue")
points(y=p$probs_gompertz, x=(p$xb), col = "green3")
points(y=p$probs_arctan, x=(p$xb), col = "orange2")
points(y=p$probs_tanh, x=(p$xb), col = "red3")



# comparing Y
gen_Y <- function(gamma, X, A, beta_Y, flavor_Y) {
  
  xb_gamma_a <- as.matrix(cbind(1,X))%*%beta_Y + gamma * A
  
  if(flavor_Y == "expo") { fun_Y = exp}
  if(flavor_Y == "square") { fun_Y = function(x) x^2}
  if(flavor_Y == "sigmoid"){ fun_Y = function(x) 1/(1+exp(-x)) * 10}
  if(flavor_Y == "nonlin") { fun_Y = function(x) (x^4) / (1 + abs(x)^3)}
  if(flavor_Y == "piece") { fun_Y = function(x) ifelse(x<0, log1p(x^2), 1/(1+exp(-x)))*6}
  if(flavor_Y == "atan") { fun_Y = function(x) 6 * (atan(x) / pi + 0.5)}
  
  Y <- fun_Y(xb_gamma_a) + abs(rnorm(n, 0, 0.01))
  
}

A <- rbinom(n, 1, p$probs_tanh)
xb_gamma_a <- as.matrix(cbind(1,X))%*%beta_Y + gamma * A
Y_sq <- gen_Y(gamma = 0.6, X=X,A=A,beta_Y, "square")
Y_expo <- gen_Y(gamma = 0.6, X=X,A=A,beta_Y, "expo")
Y_sig <- gen_Y(gamma = 0.6, X=X,A=A,beta_Y, "sigmoid")
Y_atan <- gen_Y(gamma = 0.6, X=X,A=A,beta_Y, "atan")

plot(y=Y_expo, x=xb_gamma_a, col = "black", xlim = c(-4,6), ylim = c(0,10))
points(y=Y_sq, x=xb_gamma_a, col = "blue4")
points(y=Y_sig, x=xb_gamma_a, col = "red4")
points(y=Y_atan, x=xb_gamma_a, col = "green4")


