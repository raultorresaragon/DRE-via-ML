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
n <- 360
k <- 3
p <- 8
nntype <- "dnn"
source("functions_k3plus_dnet.R")
#source("functions_k3plus.R")
source("one_sim_k3plus.R")

eps = c(120,150)
penals = c(0.001,0.005)
mytable <- tibble(dataset = numeric(),
                  estimate = character(),
                  A_01 = numeric(),
                  A_02 = numeric(),
                  A_12 = numeric())
flavor_ops <- c("tanh","sigmoid", function(x) 1/(1+exp(-x)) * 10)

# Run simulations
# ---------------
tic("all_iters")
for(i in 1:M) {
  cat(paste0("\niteration ", i))
  
  # dataset params
  rho   <- round(runif(1, 0.4, 0.6), 1)
  Xmu   <- round(runif(p, -1, 1), 1)
  beta_A <- c(1, round(runif(p, -2, 2), 1))
  beta_Y <- c(1, round(runif(p, -1, 1), 1))
  gamma <- c(0.7, 0.45)
  
  # estimation
  tic("")
  suppressWarnings(
  r <- one_sim(n = n, p = p, Xmu = Xmu, iter = i, verbose = TRUE, 
               A_flavor = flavor_ops[[1]], beta_A = beta_A, gamma = gamma, 
               Y_flavor = flavor_ops[[2]], Y_fun = flavor_ops[[3]], beta_Y = beta_Y,
               nn_eps = eps, nn_penals = penals, nntype = nntype)
  )
  toc(log = TRUE, quiet = TRUE)
  last_time <- tictoc::tic.log(format = FALSE)
  last_iter_time <-last_time[[length(last_time)]]$toc - last_time[[length(last_time)]]$tic
  cat(paste0("\n  ...run time: ", round(last_iter_time / 60, 2), " mins"))
  
  # results
  print(r$Vn_df)
  mytable <- rbind(mytable, r$my_k_row)
  tictoc::tic.clearlog()
}
total_time <- toc(log = TRUE, quiet = TRUE)
total_seconds <- total_time$toc - total_time$tic
cat(paste0("\nTotal run time: ", round(total_seconds / 60, 2), " mins"))


# Results
# -------
mytable
readr::write_csv(round(mytable, 3), 
                 paste0("tables/simk",3,"_",flavor_ops[[1]],"_",flavor_ops[[2]],".csv"))


# Save R objects
# --------------
#save.image(file="sim_01_k2.RData")
