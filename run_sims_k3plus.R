# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: double_robust.R
# Date: 2025-05-15
# Note: This script runs M simulations of
#       k=3+ treatment regime with DRE via NN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(1812)
library(tictoc)
rm(list = ls())

# Set parameters and load functions
# ---------------------------------
M <- 1
n <- 600
k <- 3
p <- 8
nntype <- "1nn"
#source("functions_k3plus_dnn.R")
source("functions_k3plus.R")
source("YAX_funs.R")
source("one_sim_k3plus.R")

eps = c(120,150)
penals = c(0.001,0.005)
hidunits = c(2L, 6L)
flavor_ops <- c("logit","expo", function(x) {exp(x)}) #tanh sigmoid 1/(1+exp(-x)) * 10


# Run simulations
# ---------------
tic("all_iters")
for(i in 1:M) {
  cat(paste0("\niteration ", i))
  
  # dataset params
  rho   <- round(runif(1, 0.4, 0.6), 1)
  Xmu   <- round(runif(p, -1, 1), 1)
  beta_A <-  cbind(c(1, round(runif(p, -2, 2),1)), 
                   c(1, round(runif(p, -2, 2),1)), 
                   c(1, round(runif(p, -2, 2),1))) |> as.matrix()
  beta_Y <- c(1, round(runif(p, -1, 1), 1))
  gamma <- c(0.7, 0.45)
  
  # estimation
  tic("")
  suppressWarnings(
  r <- one_sim(n = n, p = p, Xmu = Xmu, iter = i, k = k, verbose = TRUE, 
               A_flavor = flavor_ops[[1]], beta_A = beta_A, gamma = gamma, 
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
  
}


total_time <- toc(log = TRUE, quiet = TRUE)
total_seconds <- total_time$toc - total_time$tic
cat(paste0("\nTotal run time: ", round(total_seconds / 60, 2), " mins"))
mytable
otr_table

# Results
# -------
mytable <- 
  mytable[,c(1,2)] |> 
  cbind(
    apply(mytable[,-c(1,2)], 2, function(x) as.numeric(unname(unlist(x)))) |> 
      as.data.frame()
  )
readr::write_csv(mytable, 
                 paste0("tables/simk",k,"_",flavor_ops[[1]],"_",flavor_ops[[2]],".csv"))

readr::write_csv(otr_table, 
                 paste0("tables/OTR_simk",k,"_",flavor_ops[[1]],"_",flavor_ops[[2]],".csv"))


# Save R objects
# --------------
#save.image(file="sim_01_k2.RData")
