# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: double_robust.R
# Date: 2025-04-17
# Note: This script runs M simulations of
#       k=2 treatment regime with DRE via NN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(tictoc)

rm(list = ls())
M <- 25
n <- 1000
source("functions_k2_01.R")

amod_formula <- "A~X1+X2+X3"
ymod_formula <- "Y~X1+X2+X3"
p <- 3
rho <- 0.6
Xmu <- c(-2, 0.1, 1)
beta = c(1, seq(from = 1*.1, to = p*.1, by = .1))
gamma = 0.8
hidunits = c(5,25)
eps = c(50,200)
penals = c(0.001,0.01)
mytable <- tibble(true_model = numeric(),
                  naive_est = numeric(),
                  true_model_est = numeric(),
                  nn_model_est = numeric())

tic("simulations")
for(i in 1:M) {
  print(paste0("iteration ",i))
  r <- one_sim(n=n, p=3, Xmu=Xmu, beta=beta, gamma=gamma, 
               ymod_formula_os=ymod_formula, amod_formula_os=amod_formula,
               hidunits_os=hidunits, eps_os=eps, penals_os=penals)
  mytable <- rbind(mytable, r)
}
toc <- toc(quiet=TRUE)
print(paste0("A time:", round((toc$toc[[1]]-toc$tic[[1]])/60,2), " mins"))

mytable <- 
  mytable |>
  mutate(true_model_diff = (true_model - true_model_est),
         nn_model_diff = (true_model - nn_model_est),
         naive_model_diff = (true_model - naive_est))

mytable_long <- 
  mytable |>
  dplyr::select(ends_with("diff")) |>
  pivot_longer(cols = everything(), names_to = "estimator", values_to = "diffs") |>
  mutate(estimator = stringr::str_replace(estimator,"_diff",""))

mytitle <- "Discrepancies between true differences in means and \n model estimated differences in means"
ggplot2::ggplot(mytable_long, aes(x = estimator, y = diffs)) +
  geom_boxplot(fill = "skyblue") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.y = element_text(size = 14),
        axis.text = element_text(size = 14)) + 
  labs(title = mytitle, y = "diff", x="")

save.image(file="sim_01_k2.RData")
