# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: real_data_analysis_01.R
# Date: 2025-07-08
# Note: This script computes the OTR based on real data provided
#       by Dr. Ahn
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

library(tidyverse)
library(readr)
rm(list = ls())
set.seed(1810)
source("functions_k2_01.R")


# nn parameter space for tuning
hidunits = c(5,20)
eps = c(100,250)
penals = c(0.001,0.01)


# get data
#df <- read_csv("real_data/recoded_ASTR_t1.csv")
df <- read_csv("real_data/recoded_ASTR.csv")
Y <- df$gh
X <- cbind(df$gender, df$age, df$partial_or_total_removal) |> 
     `colnames<-`(c("X1","X2","X3"))
A <- df$chemo
dat <- as.data.frame(cbind(Y,A,X)) |> na.omit()
dim(dat)


# grab a hold out for OTR
# random_row <- round(runif(1, nrow(df)))
# dat_new <- dat[random_row,] |> dplyr::select(-A) |> `rownames<-`(NULL)
Y <- dat$Y
A <- dat$A


# specify models
pmodel <- "A~X1+X2+X3"
omodel <- "Y~X1+X2+X3"


# estimate A (propensity model)
H_logit <- glm(as.formula(pmodel), family=binomial(link="logit"), data=dat)
pscores_logit <- predict(H_logit, type = "response")
H_nn <- A_model_nn(a_func=pmodel, dat=dat, 
                   hidunits=hidunits, eps=eps, penals=penals, verbose=FALSE) 
pscores_nn <- predict(H_nn, new_data = dat %>% select(-A), type = "raw") |> as.vector()


# estimate Y (outcome model)
delta_1 <- as.numeric(A==1)
delta_0 <- as.numeric(A==0)

  ### turn expo to OLS because of Y=0 cases
g_1 <- glm(as.formula(omodel), family = gaussian(link="identity"), data = dat[A==1,])
ghat_1 <- predict(g_1, newdata = dat, type = "response")
g_0 <- glm(as.formula(omodel), family = gaussian(link="identity"), data = dat[A==0,])
ghat_0 <- predict(g_0, newdata = dat, type = "response")
muhat_1 <- mean(ghat_1 + (delta_1*(Y - ghat_1)/(pscores_logit))/(mean(delta_1/pscores_logit)))
muhat_0 <- mean(ghat_0 + (delta_0*(Y - ghat_0)/(1-pscores_logit))/(mean(delta_0/(1-pscores_logit))))
diff_means_exp <- muhat_1 - muhat_0

fit_nn <- estimate_Y_nn(dat, pi_hat=pscores_nn, ymod_formula=omodel,
                        hidunits=hidunits, eps=eps, penals=penals, 
                        verbose=FALSE)
muhat_1 <- 
  mean(fit_nn$ghat_1 + (delta_1*(Y - fit_nn$ghat_1)/(pscores_nn))/(mean(delta_1/pscores_nn)))
muhat_0 <- 
  mean(fit_nn$ghat_0 + (delta_0*(Y - fit_nn$ghat_0)/(1-pscores_nn))/(mean(delta_0/(1-pscores_nn))))
diff_means_nn <- muhat_1 - muhat_0

# compute OTR Vn
get_Vn <- function(g_1, g_0, X_new, from_model = "nn") {
  
  if(from_model=="nn"){
    V_1 <- predict(g_1, new_data = X_new, type = "raw")
    V_0 <- predict(g_0, new_data = X_new, type = "raw")
  } else {
    V_1 <- predict(g_1, newdata = X_new, type = "response")
    V_0 <- predict(g_0, newdata = X_new, type = "response")    
  }
  r <- 
    tibble(X_new) |> 
    mutate(V_1 = V_1, 
           V_0 = V_0,
           Optimal_A = if_else(max(V_1, V_0) == V_1, 1, 0))
  
}

Vn_df_logitols <- get_Vn(g_1 = g_1, 
                         g_0 = g_0, 
                         X_new = tibble(X1=1, X2=75, X3=1),
                         from_model = "logit-ols") 

Vn_df_nn <- get_Vn(g_1 = fit_nn$g_1, 
                   g_0 = fit_nn$g_0, 
                   X_new = tibble(X1=1, X2=75, X3=1),
                   from_model = "nn") 

dat$Yhat_logit_ols <- 0
dat$Yhat_nn <- 0
dat$Yhat_logit_ols[dat$A==1] <- ghat_1[dat$A==1]
dat$Yhat_logit_ols[dat$A==0] <- ghat_0[dat$A==0]
dat$Yhat_nn[dat$A==1] <- fit_nn$ghat_1[dat$A==1]
dat$Yhat_nn[dat$A==0] <- fit_nn$ghat_0[dat$A==0]


# Compute RMSE
RMSE <- function(y, yhat){
  stopifnot(length(y)==length(yhat))
  sqrt(sum((y-yhat)^2)/length(y))
}

rmse_nn <- RMSE(dat$Y, dat$Yhat_nn) |> round(1)
rmse_logit_ols <-RMSE(dat$Y, dat$Yhat_logit_ols) |> round(1)


# plot predicted Y vs actual Y in sample
mycols <- c("#CC661AB3","#33661AB3","black")
mycols <- c("black","darkgrey","blue")
legpos <- "bottomright"
cex_lab = 1.7
cex_main = 2
cex_axis = 1.7
cex_legend = 1.7

rand <- sample(1:nrow(dat), size = 150, replace = FALSE)
plot_dat <- dat[rand, ] |> arrange(Y)

jpeg("images/rwd_gh_ghhat_scatters.jpeg", width = 1273, height = 743)
par(mfrow = c(1,2))
plot(plot_dat$Y~plot_dat$Yhat_logit_ols, 
    ylab = 'GH observed', #bquote('P(A='*.(i)*")"),
    xlim = c(min(dat$Y),max(dat$Y)),
    xlab = bquote('GH predicted logit-ols (RMSE='*.(rmse_logit_ols)*")"),
    pch = c(2),
    col = c(mycols[1]),
    cex.lab = cex_lab, cex.main = cex_main, cex.axis = cex_axis)
abline(a = 0, b = 1, col = "blue", lwd = 2, lty = 2)

plot(plot_dat$Y~plot_dat$Yhat_nn, 
       ylab = 'GH observed', #bquote('P(A='*.(i)*")"),
       xlim = c(min(dat$Y),max(dat$Y)),
       xlab = bquote('GH predicted NN (RMSE='*.(rmse_nn)*")"), #bquote(hat(P)*"(A="*.(i)*")"),
       pch = c(3),
       col = c(mycols[1]),
       cex.lab = cex_lab, cex.main = cex_main, cex.axis = cex_axis)
abline(a = 0, b = 1, col = "blue", lwd = 2, lty = 2)
par(mfrow=c(1,1))
dev.off()


jpeg("images/rwd_gh_ghhat.jpeg", width = 1073, height = 743)
par(mar = c(5.1, 6, 4.1, 2.1))  
plot(sort(dat$Y), col=mycols[3],
     #main=expression("GH and " * hat(GH) * " (in sample) by model"),
     #type="b" , bty="l",
     ylab="GH",
     xlab="", #expression(X^T * hat(bold(beta))),
     cex.lab = cex_lab, cex.main = cex_main, cex.axis = cex_axis)
points(dat$Yhat_nn, col=mycols[1] , pch=3) 
points(dat$Yhat_logit_ols, col=mycols[2], pch=2)
legend("bottomright", 
       legend = c(expression(hat(GH)["nn"]), expression(hat(GH)["logit-ols"]), "Observed"), 
       col = mycols, 
       pch = c(3,2,1), 
       bty = "n", 
       pt.cex = cex_legend, 
       cex = cex_legend, 
       text.col = "black", 
       horiz = F , 
       inset = c(0.1, 0.1, 0.1))
dev.off()

#ggsave(paste0("images/rwd_gh_ghhat.jpeg"), width = 7.15, height = 4.95, dpi = 150)
# for jpeg simply multiply 7.15*150 and round up. Same with height.



