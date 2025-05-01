# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: double_robust.R
# Date: 2025-04-17
# Note: This script runs M simulations of
#       k=2 treatment regime with DRE via NN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(1234)
library(tictoc)
rm(list = ls())


# Set parameters and load functions
# ---------------------------------
M <- 10
n <- 500
source("functions_k2_01.R")

amod_formula <- "A~X1+X2+X3"
ymod_formula <- "Y~X1+X2+X3"
hidunits = c(5,25)
eps = c(50,200)
penals = c(0.001,0.01)
mytable <- tibble(prob_A = numeric(),
                  true_diff = numeric(),
                  naive_est = numeric(),
                  expo_model_est = numeric(),
                  nn_model_est = numeric())

flavor_ops <- c("logit","expo", exp)

# Run simulations
# ---------------
tic("simulations")
for(i in 1:M) {
  print(paste0("iteration ",i))
  p <- 3
  rho   <- runif(1, 0.4, 0.8)
  Xmu   <- runif(3, -2, 2)
  beta_A <- c(1, runif(3, 0.2, 0.8))
  beta_Y <- c(1, runif(3, 0.2, 0.8))
  gamma <- runif(1, 0.7, 0.9)
  r <- one_sim(n=n, p=3, Xmu=Xmu, 
               A_flavor = flavor_ops[[1]], beta_A=beta_A, gamma=gamma, 
               Y_flavor = flavor_ops[[2]], Y_fun = flavor_ops[[3]], beta_Y=beta_Y,
               ymod_formula_os=ymod_formula, amod_formula_os=amod_formula,
               nn_hidunits=hidunits, nn_eps=eps, nn_penals=penals)
  mytable <- rbind(mytable, r)
}
toc <- toc(quiet=TRUE)
cat(paste0("Total run time:", round((toc$toc[[1]]-toc$tic[[1]])/60,2), " mins"))
mytable
readr::write_csv(mytable, 
                 paste0("tables/simk2_",flavor_ops[[1]],"_",flavor_ops[[2]], ".csv"))

# Prepare output for display
# --------------------------
mytable_long <- 
  mytable |>
  mutate(expo_model_diff = (true_diff - expo_model_est),
         nn_model_diff = (true_diff - nn_model_est),
         naive_model_diff = (true_diff - naive_est)) |>
  dplyr::select(ends_with("diff")) |>
  pivot_longer(cols = everything(), names_to = "estimator", values_to = "diffs") |>
  mutate(estimator = stringr::str_replace(estimator,"_diff","")) |>
  filter(estimator != "true")

mytitle <- "Discrepancies between true differences in means and \n model estimated differences in means"
ggplot2::ggplot(mytable_long, aes(x = estimator, y = diffs)) +
  geom_boxplot(fill = "skyblue") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.y = element_text(size = 14),
        axis.text = element_text(size = 14)) + 
  labs(title = mytitle, y = "diff", x="")
ggsave(paste0("images/simk2_",flavor_ops[[1]],"_",flavor_ops[[2]], ".jpeg"),
       width = 7.15, height = 4.95, dpi = 150)

# Save R objects
# --------------
#save.image(file="sim_01_k2.RData")
