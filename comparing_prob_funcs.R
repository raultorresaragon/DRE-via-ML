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
  probs_tanh <- 0.5* (tanh(xb)+1)
  r <- list(probs_logit = probs_logit, 
            probs_tanh = probs_tanh,
            xb = xb)  
}
p <- gen_probs(X, beta_A)
plot(y=p$probs_logit, x=(p$xb), col = "lightblue3", ylim=c(0,1), xlim=c(-4,4))
points(y=p$probs_tanh, x=(p$xb), col = "darkred")

jpeg("images/logit_vs_tanh.jpeg", width = 1000, height = 750)
par(mar = c(5, 6, 4, 2))
curve(1/(1 + exp(-1*(x))), 
      from=-4, to=4, 
      lwd = 6,
      col = "skyblue3",      
      main = "Comparison of logit and tanh functions",
      ylab = "P(A=1)",
      xlab = expression(x[i]^T * beta),
      cex.lab = 2, cex.main = 2.25, cex.axis = 1.75) 
curve(0.5 * (tanh(x) + 1),
      add = TRUE,
      from=-4, to=4, 
      lwd = 6,
      col = "darkred",      
      cex.lab = 2) 
legend("bottomright", legend = c("logit", "tanh"), fill=c("skyblue3", "darkred"),
       cex=2)
dev.off()



# comparing Y
gen_Y <- function(gamma, X, A, beta_Y, flavor_Y) {
  
  xb_gamma_a <- as.matrix(cbind(1,X))%*%beta_Y + gamma * A
  
  if(flavor_Y == "expo") { fun_Y = exp}
  if(flavor_Y == "sigmoid"){ fun_Y = function(x) 1/(1+exp(-x)) * 10}
  if(flavor_Y == "gamma") { fun_Y = function(x) pgamma(x, shape = 3, rate = 2) * 10}
  if(flavor_Y == "lognormal") { fun_Y = function(x) plnorm(x, 0, 1) * 10 }
  Y <- fun_Y(xb_gamma_a) + rnorm(n, 0, 0.01)
  
}

A <- rbinom(n, 1, p$probs_tanh)
xb_gamma_a <- as.matrix(cbind(1,X))%*%beta_Y + gamma * A
Y_expo <- gen_Y(gamma = 0.6, X=X, A=A, beta_Y, "expo")
Y_sig <- gen_Y(gamma = 0.6, X=X, A=A, beta_Y, "sigmoid")
Y_gamma <- gen_Y(gamma = 0.6, X=X, A=A, beta_Y, "gamma")
Y_lognormal <- gen_Y(gamma = 0.6, X=X, A=A, beta_Y, "lognormal")
plot(y=Y_expo, x=xb_gamma_a, col = "black", xlim = c(-4,6), ylim = c(0,10))
points(y=Y_sig, x=xb_gamma_a, col = "blue4")
points(y=Y_gamma, x=xb_gamma_a, col = "red4")
points(y=Y_lognormal, x=xb_gamma_a, col = "green4")



# expo vs sigmoid  AND  logit vs rest
jpeg("images/expo_vs_sigmoid_AND_logit_vs_rest.jpeg", width = 1000, height = 750)
par(mfrow=c(1,2))

# left panel
par(mar = c(5, 6, 4, 2))
curve(1/(1 + exp(-1*(x))), 
      from=-4, to=4, 
      lwd = 6,
      col = "skyblue3",      
      main = "Propensity model",
      ylab = "P(A=1)",
      xlab = expression(x[i]^T * beta[A]),
      cex.lab = 2, cex.main = 2.25, cex.axis = 1.75) 
curve(0.5 * (tanh(x) + 1),
      add = TRUE,
      from=-4, to=4, 
      lwd = 6,
      col = "darkred",      
      cex.lab = 2) 
legend("bottomright", legend = c("logit", "tanh"), fill=c("skyblue3", "darkred"),
       cex=2)

# right panel
shape = 3; scale = 2
par(mar = c(5, 6, 4, 2))
curve(exp(x), 
      from=-3, to=3, 
      lwd = 6,
      col = "skyblue3",      
      main = "Outcome model",
      ylab = "Y",
      xlab = expression(x[i]^T * beta[Y] + gamma * A),
      cex.lab = 2, cex.main = 2.25, cex.axis = 1.75) 
curve(1/(1+exp(-x)) * 10,
      add = TRUE,
      from=-3, to=3, 
      lwd = 6,
      col = "darkred",      
      cex.lab = 2)
curve((exp(shape*x) * exp(-exp(x)/scale)) / (gamma(shape) * scale^shape) * 10,
      add = TRUE,
      from=-3, to=3, 
      lwd = 6,
      col = "#8b0046",      
      cex.lab = 2) 
curve((1 / (exp(x) * sqrt(2 * pi))) * exp(-0.5 * x^2) * 10,
      add = TRUE,
      from=-3, to=3, 
      lwd = 6,
      col = "#8b4600",      
      cex.lab = 2) 
legend("topleft", 
       legend = c("exponential", "sigmoid", "gamma", "lognormal"), 
       fill=c("skyblue3", "darkred", "#8b0046","#8b4600"),
       cex=2)
dev.off()


