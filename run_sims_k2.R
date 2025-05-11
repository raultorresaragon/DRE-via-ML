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
n <- 300
source("functions_k2_01.R")

amod_formula <- "A~X1+X2+X3"
ymod_formula <- "Y~X1+X2+X3"
hidunits = c(5,20)
eps = c(50,150)
penals = c(0.001,0.01)
mytable <- tibble(prob_A = numeric(),
                  true_diff = numeric(),
                  naive_est = numeric(),
                  expo_model_est = numeric(),
                  nn_model_est = numeric())
#nn_params_list <- 

#flavor_ops <- c("logit","expo", exp)
flavor_ops <- c("tanh","sigmoid", function(x) 1/(1+exp(-x)) * 10)

# Run simulations
# ---------------
tic("sims")
for(i in 1:M) {
  cat(paste0("\niteration ",i))
  p <- 3
  rho   <- round(runif(1, 0.4, 0.6),1)
  Xmu   <- round(runif(3, -1, 1),1)
  beta_A <- c(1, round(runif(3, -1, 1),1))
  beta_Y <- c(1, round(runif(3, -1, 1),1))
  gamma <- 0.6
  cat(paste0("\nXmu = ", paste0(Xmu, collapse=",")))
  cat(paste0("\nbeta_A = ", paste0(beta_A, collapse=",")))
  cat(paste0("\nbeta_Y = ", paste0(beta_Y, collapse=",")))
  cat(paste0("\nrho = ", rho))
  tic("one_sim")
  r <- one_sim(n=n, p=3, Xmu=Xmu, 
               A_flavor = flavor_ops[[1]], beta_A=beta_A, gamma=gamma, 
               Y_flavor = flavor_ops[[2]], Y_fun = flavor_ops[[3]], beta_Y=beta_Y,
               ymod_formula_os=ymod_formula, amod_formula_os=amod_formula,
               nn_hidunits=hidunits, nn_eps=eps, nn_penals=penals, verbose=FALSE)
  cat(paste0("\n V_1 = ", round(r$Vn$V_1,2), 
             " ; V_0 = ", round(r$Vn$V_0), 
             " ; OTR: A=", r$Vn$Optimal_A))
  cat("\n g_0_nn: ")
  r$g_0_nn |> extract_nn_params() |> print()
  cat("\n g_1_nn: ")
  r$g_1_nn |> extract_nn_params() |> print()
  toc <- toc(quiet=TRUE)
  cat(paste0("\n  ...run time: ", round((toc$toc[[1]]-toc$tic[[1]])/60,2), " mins"))  
  mytable <- rbind(mytable, r$myrow)
}
toc <- toc(quiet=TRUE)
cat(paste0("\nTotal run time:", round((toc$toc[[1]]-toc$tic[[1]])/60,2), " mins"))
mytable
readr::write_csv(round(mytable, 3), 
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
  filter(estimator != "true") |>
  #filter(diffs < quantile(diffs, probs = 0.9) & diffs > quantile(diffs,.1))|>
  mutate(estimator = factor(estimator, 
         levels = c("naive_model","expo_model","nn_model"),
         labels = c("naive", "logistic-expo", "neural network")))

mytitle <- bquote(paste("Errors between ", Delta, " and ", hat(Delta), " per model.",
                        "\n True model is ", .(flavor_ops[[1]]), "-", .(flavor_ops[[2]])))
ggplot2::ggplot(mytable_long, aes(x = estimator, y = diffs)) +
  geom_boxplot(fill = "skyblue") + 
  ylim(c(-1,1.2)) +
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
