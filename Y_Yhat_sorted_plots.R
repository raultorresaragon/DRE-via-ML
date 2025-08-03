# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: predicted_A_Y_plots_k3
# Date: 2025-07-31
# Note: This script saves 6-plot image of predicted A(0,1,2) and Y(0) Y(1) Y(2) by model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


plot_predicted_A_Y <-function(beta_A, beta_Y, dat, 
                              fit_Y_nn, fit_Y_expo, gamma, 
                              fit_A_nn, fit_A_logit, A_flavor, Y_flavor, ds, k){
  
  jpeg(paste0("images/YYhat_sorted_k", k, A_flavor, Y_flavor, "_dset", ds, ".jpeg"), 
       width = 1000, height = 510)
  
  Y <- dat$Y
  X <- dplyr::select(dat, starts_with("X"))
  A <- dat$A
  
  par(mfrow = c(2,k), mar = c(5.1, 5.8, 4.1, 1.3))
  #                                   True   logit/expo  NN     True
  mycols = mycolors <- adjustcolor(c("black","#FB8072","blue"), alpha.f = 0.9)
  xb_A <-(as.matrix(cbind(1,X))%*%beta_A)
  b = 1
  legpos <- "topleft"
  cex_lab = 1.4
  cex_main = 1.8
  cex_axis = 1.4
  cex_legend = 1.4
  sample <- sample(1:floor(length(Y)/k), 200)
  
  if(A_flavor=="tanh") { 
    plt_p <- b * 0.5* (tanh(xb_A)+1)
    curve_A <- function(x) { (b * 0.5* (tanh(x)+1))}
  }
  if(A_flavor=="logit"){ 
    plt_p <- b * 1/(1 + exp(-1*(xb_A)))
    curve_A <- function(x) { (b * 1/(1 + exp(-1*(x)))) }
  }
  
  colnames(plt_p) <- paste0("true_pscores",0:(ncol(plt_p)-1))
  dat <- cbind(dat, plt_p)
  plt_p <- plt_p/rowSums(plt_p)
  Yhat_nn <- rep(NA, length(Y))
  Yhat_expo <- rep(NA, length(Y))
  Yhat_nn[A==0] <- fit_Y_nn$A_01[[4]][A==0]
  Yhat_expo[A==0] <- fit_Y_expo$A_01[[4]][A==0]
  for(i in 1:(k-1)) {
    Yhat_nn[A==i] <- fit_Y_nn[[paste0("A_0",i)]][[5]][A==i]
    Yhat_expo[A==i] <- fit_Y_expo[[paste0("A_0",i)]][[5]][A==i]
  }
  
  colnames(fit_A_logit$pscores) <- paste0(colnames(fit_A_logit$pscores), "_logit")
  colnames(fit_A_nn$pscores) <- paste0(colnames(fit_A_nn$pscores), "_nn")
  dat <- cbind(dat, fit_A_logit$pscores)
  dat <- cbind(dat, fit_A_nn$pscores)
  dat$Yhat_nn <- Yhat_nn
  dat$Yhat_expo <- Yhat_expo
  
  # P(A=i) vs. \hat{P}(A=i)
  for(i in 0:(k-1)){
    P = i
    plot(sort(dat[[paste0("true_pscores",i)]]), pch=1, col=mycols[1],
         #main = paste0("P(A=",i,")"),
         ylab=paste0("true pscore for P(A=",i,")"),
         xlab=paste0("predicted pscore for P(A=",i,")"),
         cex.lab = cex_lab, cex.main = cex_main, cex.axis = cex_axis)
    points(dat[order(dat[[paste0("true_pscores",i)]]),paste0("pscores_",i,"_logit")],
           col=mycols[2], pch=2)
    points(dat[order(dat[[paste0("true_pscores",i)]]),paste0("pscores_",i,"_nn")],
           col=mycols[3], pch=3)

    #plot((plt_p[,i+1][sample]~fit_A_logit$pscores[[paste0("pscores_",i,"_logit")]][sample]), col="black",
    #     ylab="pscore",
    #     xlab="predcited pscore",
    #     cex.lab = cex_lab, cex.main = cex_main, cex.axis = cex_axis)
    #points(fit_A_logit$pscores[[paste0("pscores_",i,"_logit")]][sample], col=mycols[2], pch=2)
    #points(fit_A_nn$pscores[[paste0("pscores_",i,"_nn")]][sample]   , col=mycols[3], pch=3) 
    #abline(a = 0, b = 1, col = "blue", lwd = 2, lty = 2)
    if(i==0){
      legend(legpos, 
             legend = c("true", "logit", "nn"), 
             col=mycols,
             pch=1:3,
             cex=cex_legend)
    }
  }
  
  ## Y vs \hat{Y}
  for(d in 0:(k-1)){
    d_j <- A==d #rep(TRUE, length(A))
    plot(sort(dat$Y[d_j]), pch=1, col=mycols[1],
         #main=bquote(Y["A="*.(d)]),
         ylab=bquote("observed " * Y["A="*.(d)]),
         xlab=bquote("predicted " * Y["A="*.(d)]),
         cex.lab = cex_lab, cex.main = cex_main, cex.axis = cex_axis)
    points(dat[order(dat$Y),"Yhat_expo"][d_j],
           col=mycols[2], pch=2)
    points(dat[order(dat$Y),"Yhat_nn"][d_j],
           col=mycols[3], pch=3)
    
    
    #plot(sort(Y[d_j]), col=mycols[3],
    #     ylab="Y",
    #     xlab=expression(hat(Y)),
    #     cex.lab = cex_lab, cex.main = cex_main, cex.axis = cex_axis)
    #points(Yhat_expo[d_j], col=mycols[2], pch=2)
    #points(Yhat_nn[d_j]  , col=mycols[1], pch=3) 
    #abline(a = 0, b = 1, col = "blue", lwd = 2, lty = 2)
    if(d==0){
      legend(legpos, 
             legend = c("observed","expo","nn"), 
             col=mycols,
             pch=1:3,
             cex=cex_legend)
    }
  }
  
  par(mfrow = c(1,1))
  dev.off()
  
}