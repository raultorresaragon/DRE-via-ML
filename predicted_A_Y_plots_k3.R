# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: predicted_A_Y_plots_k3
# Date: 2025-07-17
# Note: This script saves 6-plot image of predicted A(0,1,2) and Y(0) Y(1) Y(2) by model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


plot_predicted_A_Y <-function(beta_A, beta_Y, Y, X, A, 
                              fit_Y_nn, fit_Y_expo, gamma, 
                              fit_A_nn, fit_A_logit, A_flavor, Y_flavor, ds, k){
  
  jpeg(paste0("images/predicted_A_Y_k", k, A_flavor, Y_flavor, "_dset", ds, ".jpeg"), 
       width = 1000, height = 510)
  
  
  par(mfrow = c(2,k), mar = c(5.1, 5.8, 4.1, 1.3))
  mycolors = mycolors <- adjustcolor(c("#FB8072", "#80B1D3"), alpha.f = 0.9)
  
  gamma_ <- c(0,gamma)
  xb_A <-(as.matrix(cbind(1,X))%*%beta_A) 
  xb_Y <-(as.matrix(cbind(1,X))%*%beta_Y) 
  b <- 1 #1/k
  if(A_flavor=="tanh") { 
    plt_p <- b * 0.5* (tanh(xb_A)+1)
    curve_A <- function(x) { (b * 0.5* (tanh(x)+1)) }
  }
  if(A_flavor=="logit"){ 
    plt_p <- b * 1/(1 + exp(-1*(xb_A)))
    curve_A <- function(x) { (b * 1/(1 + exp(-1*(x)))) }
  }
  if(Y_flavor=="sigmoid") { 
    curve_Y <- lapply(seq_along(gamma_), function(g) {
      function(x) 1/(1+exp(-x-gamma_[g])) * 10
    })
    legpos <- "bottomright"
  }
  if(Y_flavor=="expo"){
    curve_Y <- lapply(seq_along(gamma_), function(g) {
      function(x) exp(x + gamma_[g])
    })
    legpos <- "topleft"
  }
  
  Yhat_nn <- rep(NA, length(Y))
  Yhat_expo <- rep(NA, length(Y))
  Yhat_nn[A==0] <- fit_Y_nn$A_01[[4]][A==0]
  Yhat_expo[A==0] <- fit_Y_expo$A_01[[4]][A==0]
  for(i in 1:(k-1)) {
    Yhat_nn[A==i] <- fit_Y_nn[[paste0("A_0",i)]][[5]][A==i]
    Yhat_expo[A==i] <- fit_Y_expo[[paste0("A_0",i)]][[5]][A==i]
  }

  for(i in 0:(k-1)){
  ## plotting A P(A=0), P(A=1), ... P(A=k-1)
  plot(plt_p[,i+1]~xb_A[,i+1], lwd=1, cex=1, 
       main=bquote("P(A=" * .(i) * ") and " * hat(P) * "(A=" * .(i) * ")"),
       ylab = bquote("P(A="*.(i)*")"),
       xlab = expression(x[i]^T * beta[A]),
       xlim = c(min(xb_A), max(xb_A)),
       cex.lab = 2, cex.main = 2.25, cex.axis = 1.75)
  curve(expr = curve_A, 
        from=min(xb_A[,i+1]), to=max(xb_A[,i+1]), 
        lwd = 2,
        col = "black",      
        add = TRUE) 
  points((b*fit_A_logit$pscores[[paste0("pscores_",i)]])~xb_A[,i+1], cex=1.5, col=mycolors[1], pch = 2)
  points((b*fit_A_nn$pscores[[paste0("pscores_",i)]])~xb_A[,i+1], cex=1.5, col=mycolors[2], pch = 3)
  if(i==0){
    legend("topleft", 
           legend = c("true","logit","nn"), 
           col=c("black", mycolors[1], mycolors[2]),
           pch=c(1,2,3),
           cex=1.9)
  }
  }
  
  ## plotting Y[A==i]
  for(d in 0:(k-1)){
    d_j <- A==d
    curve_Ya <- curve_Y[[d+1]]
    plot(Y[d_j]~xb_Y[d_j], 
         main=bquote("Y and "*hat(Y) * " for A=" * .(d)),
         ylab = "Y",
         cex.lab = 2, cex.main = 2.25, cex.axis = 1.75,
         xlab = bquote(x[i]^T * beta[Y] * "+" * gamma[.(d)]))
    curve(curve_Ya, 
          from=min(xb_Y[d_j]), to=max(xb_Y[d_j]), 
          lwd = 2,
          col = "black",      
          add = TRUE)
    points(Yhat_expo[d_j]~xb_Y[d_j], cex=1.5, col=mycolors[1], pch=2)
    points(Yhat_nn[d_j]~xb_Y[d_j], cex=1.5, col=mycolors[2], pch=3)
    if(d==0){
      legend(legpos, 
           legend = c("true","expo","nn"), 
           col=c("black", mycolors[1], mycolors[2]),
           pch = c(1,2,3),
           cex=1.9)
    }
  }
  
  par(mfrow = c(1,1))
  dev.off()
  
}


