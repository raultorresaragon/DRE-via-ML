# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: double_robust.R
# Date: 2025-05-15
# Note: This script runs M simulations of
#       k=3+ treatment regime with DRE via NN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(1857)
library(tictoc)
rm(list = ls())
par(mfrow=c(1,1))

# Set parameters and load functions
# ---------------------------------
export_tables <- TRUE
export_images <- TRUE
zero_effect = FALSE
root = paste0(getwd(),"/_", as.numeric(!zero_effect), "trt_effect/")
M <- 10
K <- c(2,3,5)   
Y_param = "ols" #"expo"
pflavs <- c("l","t")
oflavs <- c("e","s","l","g")
flavors <- #pairwise combination of flavors
  tidyr::expand_grid(pflavs, oflavs) |> 
  dplyr::mutate(l = paste0(pflavs, oflavs)) |> 
  dplyr::pull("l")  
if(length(flavors)==8) flavors <- flavors[c(1,6,7,8)]

source("YAX_funs.R")
source("outcome_models.R")
source("pscores_models.R")
source("get_diff.R")
source("get_true_diff.R")
source("compute_Vn.R")
source("one_sim_k3plus.R")
source("Y_Yhat_sorted_plots.R")

for(k in K) {
  if(k==2){ p<-3}
  if(k==3){ p<-8}
  if(k==5){ p<-12}
  n <- k*300
  eps = c(120,180)
  penals = c(0.001,0.005)
  hidunits = c(2L, 8L)
  flavor_ops <- NULL


  # Run simulations
  # ---------------
  tic("all_iters")
  for(flav in flavors) {
  
    # Iterate over DGP flavor
    if(flav == "le") flavor_ops <- c("logit","expo", 1, 0.5) 
    if(flav == "ls") flavor_ops <- c("logit","sigmoid", 1, 1) 
    if(flav == "ll") flavor_ops <- c("logit","lognormal", 1, 1)
    if(flav == "lg") flavor_ops <- c("logit","gamma", 1, 1)
    
    if(flav == "te") flavor_ops <- c("tanh", "expo", 1, 0.5)
    if(flav == "ts") flavor_ops <- c("tanh", "sigmoid", 1, 1)
    if(flav == "tl") flavor_ops <- c("tanh", "lognormal", 1, 1)
    if(flav == "tg") flavor_ops <- c("tanh", "gamma", 1, 1)
  
    # Iterate over number of datasets
    for(i in 1:M) {
      cat(paste0("\niteration ", i))
  
      # dataset params
      rho   <- round(runif(1, 0.4, 0.6), 1)
      Xmu   <- round(runif(p, -1, 1), 1)
      beta_A <-  
        matrix(rep(1,(k-1)), nrow=1) |> 
        rbind(matrix(round(runif(p*(k-1), -2, 2),1), nrow=p))
      beta_Y <- c(1, round(runif(p, -1, 1), 1)) * as.numeric(flavor_ops[[4]])
      gamma <- c(0.6, 0.4, 0.75, 0.17)[1:(k-1)] * as.numeric(!zero_effect)

      # estimation
      tic("")
      suppressWarnings(
      r <- one_sim(n = n, p = p, Xmu = Xmu, iter = i, k = k, verbose = TRUE, 
                   A_flavor = flavor_ops[[1]], beta_A = beta_A, gamma = gamma[1:(k-1)], 
                   Y_flavor = flavor_ops[[2]], beta_Y = beta_Y,
                   Y_param = Y_param,
                   hidunits = hidunits, eps = eps, penals = penals, 
                   export_images = export_images)
      )
      toc(log = TRUE, quiet = TRUE)
      last_time <- tictoc::tic.log(format = FALSE)
      last_iter_time <-last_time[[length(last_time)]]$toc - last_time[[length(last_time)]]$tic
      cat(paste0("\n  ...run time: ", round(last_iter_time / 60, 2), " mins"))
  
      # results
      print(r$Vn_df)
      if(i==1) {
          mytable <- r$my_k_row
          otr_table <- cbind(r$Xnew_Vn, r$Vn_df)
      } else {
          mytable <- rbind(mytable, r$my_k_row)
          otr_table <- rbind(otr_table, cbind(r$Xnew_Vn, r$Vn_df))
      }
      otr_table <- otr_table |> dplyr::select(dataset, OTR, contains("X"), contains("g"))
      mytable
      otr_table
      tictoc::tic.clearlog()
    } # close for M

    total_time <- toc(log = TRUE, quiet = TRUE)
    total_seconds <- total_time$toc - total_time$tic
    cat(paste0("\nTotal run time: ", round(total_seconds / 60, 2), " mins"))
    mytable
    otr_table
    print(paste0("For k=",k, "_", flavor_ops[[1]],"_",flavor_ops[[2]]))
    print(mytable)

    # Results
    # -------
    if(export_tables == TRUE){
      if(k>2) {
        mytable <- 
          mytable[,c(1,2)] |> 
          cbind(
            apply(mytable[,-c(1,2)], 2, function(x) as.numeric(unname(unlist(x)))) |> 
            as.data.frame()
          )
      }else {
        colnames(mytable) <- c("dataset", "estimate", "A_01", "pval")
        mytable$A_01 <- as.numeric(unname(unlist(mytable$A_01)))
        mytable$pval <- as.numeric(unname(unlist(mytable$pval)))
      }
      readr::write_csv(mytable, 
               paste0(root,"tables/logit_",Y_param,
                      "/simk",k,"_",flavor_ops[[1]],"_",flavor_ops[[2]],".csv"))
      
      readr::write_csv(otr_table, 
               paste0(root,"tables/logit_",Y_param,
                      "/OTR_simk",k,"_",flavor_ops[[1]],"_",flavor_ops[[2]],".csv"))
      
      #Generate LaTeX for a table: 
      cat("\n\n")
      print("LaTeX code")
      print(paste0("Estimates table for k=",k, "_", flavor_ops[[1]],"_",flavor_ops[[2]]))
      print(xtable::xtable(mytable, include.rownames = FALSE))
      ##cat("\n\n")
      ##print(paste0("OTR table for k=",k, "_", flavor_ops[[1]],"_",flavor_ops[[2]]))
      ##print(xtable::xtable(otr_table, include.rownames = FALSE))
      ##cat("\n\n")
    }
  } # close for flav
} # close for k


