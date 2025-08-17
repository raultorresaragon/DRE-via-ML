# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: predicted_A_Y_plots_k3
# Date: 2025-07-31
# Note: This script saves 6-plot image of predicted A(0,1,2) and Y(0) Y(1) Y(2) by model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


plot_predicted_A_Y <-function(beta_A, beta_Y, dat, 
                              fit_Y_nn, fit_Y_expo, gamma, 
                              fit_A_nn, fit_A_logit, A_flavor, Y_flavor, ds, k, save=TRUE, 
                              blue = TRUE){
  if(save==TRUE){
    jpeg(paste0("images/YYhat_sorted_k", 
         k, A_flavor, Y_flavor, "_dset", ds, ".jpeg"), width = 1000, height = 510)
  }
  
  Y <- dat$Y
  X <- dplyr::select(dat, starts_with("X"))
  A <- dat$A
  
  par(mfrow = c(2,(k)), mar = c(5.1, 5.8, 4.1, 1.3))
  #                                    True    logit/expo  NN    
  #mycols = mycolors <- adjustcolor(c("black","#FB8072","blue"), alpha.f = 0.9)
  if(blue == TRUE){
    mycols = mycolors <- adjustcolor(c("black","darkgray","blue"), alpha.f = 0.9)
  } else {
    mycols = mycolors <- adjustcolor(c("black","darkgray","darkgray"), alpha.f = 0.9)
  }
  xb_A <-(as.matrix(cbind(1,X))%*%beta_A)
  legposA <- "topleft"
  cex_lab = 1.4
  cex_main = 1.8
  cex_axis = 1.4
  cex_legend = 1.4
  sample <- sample(1:floor(length(Y)), k*100, replace=FALSE)
  
  ## Add true propensity scores to dat
  if(A_flavor == "tanh"){
    #xb_A <- cbind(xb_A, 0) # adding the baseline class (a=0)
    xb <- (as.matrix(cbind(1,X))%*%beta_A) 
    raw_scores <- 
      as.data.frame(0.5 * (tanh(xb)+1)) |> 
      mutate(dummyzero1=0, dummyzero2=0) # add zero so rowSums works with k=3
    
    probs <- data.frame(class1 = rep(NA, nrow(X)))
    for(i in 1:ncol(xb)) {
      probs[[paste0("class",i)]] <- raw_scores[,i] / (1 + rowSums(raw_scores[,-i]))
    }
    if(k==2) {
      sum_all_other_classes <- 1 - probs$class1
    } else {
      sum_all_other_classes <- 1 - rowSums(cbind(probs[,1:ncol(probs)], rep(0,nrow(probs))))
    }
    probs[[paste0("class",k)]] <- sum_all_other_classes |> as.vector()

    if(k==2){
      colnames(probs) <- c("true_pscores1","true_pscores0")
    } else {
      colnames(probs) <- paste0("true_pscores",0:(ncol(probs)-1))
    }
    legposY <- "bottomright"
  }
  
  if(A_flavor=="logit"){ 
    exp_xb_A <- exp(xb_A)
    denom <- 1 + rowSums(exp_xb_A)
    probs <- 1/denom
    for(i in 1:dim(beta_A)[2]) {
      probs <- cbind(probs, exp_xb_A[,i]/denom)
    }
    if(k==2){
      colnames(probs) <- c("true_pscores1","true_pscores0")
    } else {
      colnames(probs) <- paste0("true_pscores",0:(ncol(probs)-1))
    }
    legposY <- "topleft"
  }
  dat <- cbind(dat, probs)
  
  
  ## Add predicted pscores A
  colnames(fit_A_logit$pscores) <- paste0(colnames(fit_A_logit$pscores), "_logit")
  colnames(fit_A_nn$pscores) <- paste0(colnames(fit_A_nn$pscores), "_nn")
  dat <- cbind(dat, fit_A_logit$pscores)
  dat <- cbind(dat, fit_A_nn$pscores)
  if(k==2){
    dat$pscores_0_logit <- 1-dat$pscores_1_logit
    dat$pscores_0_nn <- 1-dat$pscores_1_nn
  }
  
  ## Add predicted outcome Y values
  Yhat_nn <- rep(NA, length(Y))
  Yhat_expo <- rep(NA, length(Y))
  Yhat_nn[A==0] <- fit_Y_nn$A_01[[4]][A==0]
  Yhat_expo[A==0] <- fit_Y_expo$A_01[[4]][A==0]
  for(i in 1:(k-1)) {
    Yhat_nn[A==i] <- fit_Y_nn[[paste0("A_0",i)]][[5]][A==i]
    Yhat_expo[A==i] <- fit_Y_expo[[paste0("A_0",i)]][[5]][A==i]
  }
  dat$Yhat_nn <- Yhat_nn
  dat$Yhat_expo <- Yhat_expo
  
  dat <- dat[sample,]
  dat_pA <- dat[sample(1:length(sample), size=100, replace = FALSE), ]
  
  ## Propensity score A plots
  l=0
  for(i in 0:(k-1)){
    l = l+1
    plot(sort(dat_pA[[paste0("true_pscores",i)]]), type="l", lwd=2, col=mycols[1],
         #main = paste0("P(A=",i,")"),
         ylab=paste0("true pscore for P(A=",i,")"),
         xlab=paste0("predicted pscore for P(A=",i,")"),
         cex.lab = cex_lab, cex.main = cex_main, cex.axis = cex_axis)
    points(dat_pA[order(dat_pA[[paste0("true_pscores",i)]]),paste0("pscores_",i,"_logit")],
           col=mycols[2], pch=2)
    points(dat_pA[order(dat_pA[[paste0("true_pscores",i)]]),paste0("pscores_",i,"_nn")],
           col=mycols[3], pch=3)
    if(l==1){ #<-display legend in the first plot only
      legend(legposA, 
             legend = c("true", "logit", "nn"), 
             col=mycols,
             lty=c(1,NA,NA),
             pch=c(NA,2,3),
             cex=cex_legend)
    }
  }
  
  ## Outcome Y plots
  for(d in 0:(k-1)){
    d_j <- A==d #rep(TRUE, length(A))
    plot(sort(dat$Y[d_j]), type="l", lwd=2, col=mycols[1],
         ylab=bquote("observed " * Y["A="*.(d)]),
         xlab=bquote("predicted " * Y["A="*.(d)]),
         cex.lab = cex_lab, cex.main = cex_main, cex.axis = cex_axis)
    points(dat[order(dat$Y),"Yhat_expo"][d_j],
           col=mycols[2], pch=2)
    points(dat[order(dat$Y),"Yhat_nn"][d_j],
           col=mycols[3], pch=3)
    if(d==0){ #<-display legend in the first plot only
      legend(legposY, 
             legend = c("observed","expo","nn"), 
             col=mycols,
             lty=c(1,NA,NA),
             pch=c(NA,2,3),
             cex=cex_legend)
    }
  }
  
  par(mfrow = c(1,1))
  if(save==TRUE){
    dev.off()
  }
  
}
