# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: predicted_A_Y_plots_k3
# Date: 2025-07-17
# Note: This script saves 6-plot image of predicted A(0,1,2) and Y(0) Y(1) Y(2) by model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


plot_predicted_A_Y <-function(beta_A, beta_Y, Y, X, A, 
                              fit_Y_nn, fit_Y_expo, gamma, 
                              fit_A_nn, fit_A_logit, A_flavor, Y_flavor, ds, k){
  
  jpeg(paste0("images/AY_vs_predicted_AY_k", k, A_flavor, Y_flavor, "_dset", ds, ".jpeg"), 
       width = 1000, height = 510)
  
  par(mfrow = c(2,k), mar = c(5.1, 5.8, 4.1, 1.3))
  mycolors = mycolors <- adjustcolor(c("#FB8072", "#80B1D3"), alpha.f = 0.9)
  mycolors = mycolors <- adjustcolor(c("black", "darkgray"), alpha.f = 0.9)
  xb_A <-(as.matrix(cbind(1,X))%*%beta_A)
  b = 1
  legpos <- "bottomright"
  cex_lab = 1.6
  cex_main = 2
  cex_axis = 1.6
  cex_legend = 1.6
  
  if(A_flavor=="tanh") { 
    plt_p <- b * 0.5* (tanh(xb_A)+1)
    curve_A <- function(x) { (b * 0.5* (tanh(x)+1)) }
  }
  if(A_flavor=="logit"){ 
    plt_p <- b * 1/(1 + exp(-1*(xb_A)))
    curve_A <- function(x) { (b * 1/(1 + exp(-1*(x)))) }
  }
  
  Yhat_nn <- rep(NA, length(Y))
  Yhat_expo <- rep(NA, length(Y))
  Yhat_nn[A==0] <- fit_Y_nn$A_01[[4]][A==0]
  Yhat_expo[A==0] <- fit_Y_expo$A_01[[4]][A==0]
  for(i in 1:(k-1)) {
    Yhat_nn[A==i] <- fit_Y_nn[[paste0("A_0",i)]][[5]][A==i]
    Yhat_expo[A==i] <- fit_Y_expo[[paste0("A_0",i)]][[5]][A==i]
  }
  
  # P(A=i) vs. \hat{P}(A=i)
  for(i in 0:(k-1)){
    P = i
    plot(plt_p[,i+1]~fit_A_logit$pscores[[paste0("pscores_",i)]], 
         #main=bquote("P(A=" * .(i) * ")," *hat(P) * "(A=" * .(i) * ")"),
         main=bquote("P(A=" * .(i) * ")"),
         ylab = 'observed', #bquote('P(A='*.(i)*")"),
         xlab = 'predicted', #bquote(hat(P)*"(A="*.(i)*")"),
         pch = c(2),
         col = c(mycolors[1]),
         cex.lab = cex_lab, cex.main = cex_main, cex.axis = cex_axis)
    points(plt_p[,i+1]~fit_A_nn$pscores[[paste0("pscores_",i)]], 
         pch = c(3),
         col = c(mycolors[2]))
    abline(a = 0, b = 1, col = "blue", lwd = 2, lty = 2)
    if(i==0){
      legend(legpos, 
             legend = c("logit","nn"), 
             col=c(mycolors[1], mycolors[2]),
             pch=c(2,3),
             cex=cex_legend)
    }
  }
  
  ## Y vs \hat{Y}
  for(d in 0:(k-1)){
    d_j <- A==d
    plot(Y[d_j]~Yhat_expo[d_j], 
        #main=bquote("Y," *hat(Y) * " for A=" * .(d)),
        main=bquote(Y["A="*.(d)]),
        ylab = 'observed', #bquote(Y["A="*.(d)]),
        xlab = 'predicted', #bquote(hat(Y)["A="*.(d)]),
        #xlab = bquote(x[i]^T * beta[Y] * "+" * gamma[.(d)]))
        pch = 2, 
        col = c(mycolors[1]),
        cex.lab = cex_lab, cex.main = cex_main, cex.axis = cex_axis)
    points(Y[d_j]~Yhat_nn[d_j], 
        pch = 3, 
        col = c(mycolors[2]))
    abline(a = 0, b = 1, col = "blue", lwd = 2, lty = 2)
    if(d==0){
      legend(legpos, 
             legend = c("expo","nn"), 
             col=c(mycolors[1], mycolors[2]),
             pch = c(2,3),
             cex=cex_legend)
    }
  }
  
  par(mfrow = c(1,1))
  dev.off()
  
}


