# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: double_robust.R
# Date: 2025-05-15
# Note: This script runs M simulations of
#       k=3+ treatment regime with DRE via NN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(1609)
library(tictoc)
rm(list = ls())
par(mfrow=c(1,1))

# Set parameters and load functions
# ---------------------------------
export_tables <- FALSE
M <- 2
K <- c(2,3,5)      #c(2,3,5)
flavors <- "le"  #c("ts","le")
source("functions_k3plus.R")
source("YAX_funs.R")
source("one_sim_k3plus.R")
source("Y_Yhat_sorted_plots.R")

for(k in K) {
  if(k==2){ p<-3}
  if(k==3){ p<-8}
  if(k==5){ p<-10}
  n <- k*350
  nntype <- "1nn"
  eps = c(120,150)
  penals = c(0.001,0.005)
  hidunits = c(2L, 6L)
  flavor_ops <- NULL


  # Run simulations
  # ---------------
  tic("all_iters")
  for(flav in flavors) {
  
    # Iterate over DGP flavor
    if(flav == "le") {
      flavor_ops <- c("logit","expo", function(x) {exp(x)}, 3, 0.5) 
    } else {
      flavor_ops <- c("tanh","sigmoid", function(x) {1/(1+exp(-x)) * 10}, 1, 1) 
    }
  
    # Iterate over number of dasets
    for(i in 1:M) {
      cat(paste0("\niteration ", i))
  
      # dataset params
      rho   <- round(runif(1, 0.4, 0.6), 1)
      Xmu   <- round(runif(p, -1, 1), 1)
      beta_A <-  
        matrix(rep(1,(k-1)), nrow=1) |> 
        rbind(matrix(round(runif(p*(k-1), -2, 2),1), nrow=p))
      beta_Y <- c(1, round(runif(p, -1, 1), 1)) * flavor_ops[[5]]
      gamma <- c(0.8, 0.6, 0.52, 0.37)[1:(k-1)] * flavor_ops[[4]]
  
      # estimation
      tic("")
      suppressWarnings(
      r <- one_sim(n = n, p = p, Xmu = Xmu, iter = i, k = k, verbose = FALSE, 
                   A_flavor = flavor_ops[[1]], beta_A = beta_A, gamma = gamma[1:(k-1)], 
                   Y_flavor = flavor_ops[[2]], Y_fun = flavor_ops[[3]], beta_Y = beta_Y,
                   hidunits = hidunits, eps = eps, penals = penals, nntype = nntype)
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
        colnames(mytable) <- c("dataset", "estimate", "A_01")
        mytable$A_01 <- as.numeric(unname(unlist(mytable$A_01)))
      }
      readr::write_csv(mytable, 
                     paste0("tables/simk",k,"_",flavor_ops[[1]],"_",flavor_ops[[2]],".csv"))
      
      readr::write_csv(otr_table, 
                     paste0("tables/OTR_simk",k,"_",flavor_ops[[1]],"_",flavor_ops[[2]],".csv"))
      
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


