# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: predicted_A_Y_plots_k2
# Date: 2025-07-17
# Note: This script saves 3-plot image of predicted A and Y(0) Y(1) by model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


plot_predicted_A_Y <-function(beta_A, beta_Y, Y, X, A, 
                              fit_nn, fit_expo, gamma, 
                              pscores_logit, pscores_nn, A_flavor, Y_flavor, ds, k){

  jpeg(paste0("images/predicted_A_Y_k", k, A_flavor, Y_flavor, "_dset", ds, ".jpeg"), 
       width = 1000, height = 510)
  par(mfrow = c(1,3))
  mycolors = c("#FB8072","#80B1D3")
  
  xb_A <-(as.matrix(cbind(1,X))%*%beta_A) 
  xb_Y <-(as.matrix(cbind(1,X))%*%beta_Y) 
  if(A_flavor=="tanh") { 
    plt_p <- 0.5* (tanh(xb_A)+1)
    curve_A <- function(x) { (0.5* (tanh(x)+1)) }
  }
  if(A_flavor=="logit"){ 
    plt_p <- 1/(1 + exp(-1*(xb_A)))
    curve_A <- function(x) { (1/(1 + exp(-1*(x)))) }
  }
  if(Y_flavor=="sigmoid") { 
    curve_Ya1 <- function(x) {1/(1+exp(-x-gamma)) * 10}
    curve_Ya0 <- function(x) {(1/(1+exp(-x-0)) * 10)}
  }
  if(Y_flavor=="expo"){ 
    curve_Ya1 <- function(x) {exp(x+gamma)}
    curve_Ya0 <- function(x) {exp(x+0)}
  }
  
  Yhat_nn <- rep(NA, length(Y))
  Yhat_nn[A==1] <- fit_nn$ghat_1[A==1]
  Yhat_nn[A==0] <- fit_nn$ghat_0[A==0]
  Yhat_expo <- rep(NA, length(Y))
  Yhat_expo[A==1] <- fit_expo$ghat_1[A==1]
  Yhat_expo[A==0] <- fit_expo$ghat_0[A==0]  
  
  
  ## plotting A
  plot(plt_p~xb_A, lwd=1, cex=1, 
       main=expression("P(A=1) and "*hat(P)*"(A=1)"),
       ylab = "P(A=1)",
       xlab = expression(x[i]^T * beta[A]),
       xlim = c(min(xb_A), max(xb_A)),
       cex.lab = 2, cex.main = 2.25, cex.axis = 1.75)
  curve(expr = curve_A, 
        from=min(xb_A), to=max(xb_A), 
        lwd = 2,
        col = "black",      
        add = TRUE) 
  points(pscores_logit~xb_A, cex=1.5, col=mycolors[1], pch = 2)
  points(pscores_nn~xb_A, cex=1.5, col=mycolors[2], pch = 3)
  legend("bottomright", 
         legend = c("true","logit","nn"), 
         col=c("black", mycolors[1], mycolors[2]),
         pch=c(1,2,3),
         cex=1.9)
  
  
  ## plotting Y[A==1]
  for(d in c(0,1)){
    d_j <- A==d
    if(d==0){
      curve_Y <- curve_Ya0
    } else { 
      curve_Y <- curve_Ya1
    }
    plot(Y[d_j]~xb_Y[d_j], 
         main=bquote("Y and "*hat(Y) * " for A=" * .(d)),
         ylab = "Y",
         cex.lab = 2, cex.main = 2.25, cex.axis = 1.75,
         xlab = bquote(x[i]^T * beta[Y] * "+" * gamma))
    curve(curve_Y, 
          from=min(xb_Y[d_j]), to=max(xb_Y[d_j]), 
          lwd = 2,
          col = "black",      
          add = TRUE)
    points(Yhat_expo[d_j]~xb_Y[d_j], cex=1.5, col=mycolors[1], pch=2)
    points(Yhat_nn[d_j]~xb_Y[d_j], cex=1.5, col=mycolors[2], pch=3)
    legend("bottomright", 
           legend = c("true","logit","nn"), 
           col=c("black", mycolors[1], mycolors[2]),
           pch = c(1,2,3),
           cex=1.9)
  }
  dev.off()
  
  
}


