library(tidyverse)
library(ggplot2)
library(patchwork)

rm(list = ls())
simk2_logit_expo <- read_csv("tables/simk2_logit_expo.csv")
simk3_logit_expo <- read_csv("tables/simk3_logit_expo.csv")
#simk5_logit_expo <- read_csv("tables/simk5_logit_expo.csv")

simk2_tanh_sigmoid <- read_csv("tables/simk2_tanh_sigmoid.csv")
simk3_tanh_sigmoid <- read_csv("tables/simk3_tanh_sigmoid.csv")
#simk5_tanh_sigmoid <- read_csv("tables/simk5_tanh_sigmoid.csv")

simk2_tanh_lognormal <- read_csv("tables/simk2_tanh_lognormal.csv")
simk3_tanh_lognormal <- read_csv("tables/simk3_tanh_lognormal.csv")
#simk5_tanh_lognormal <- read_csv("tables/simk5_tanh_lognormal.csv")

simk2_tanh_gamma <- read_csv("tables/simk2_tanh_gamma.csv")
simk3_tanh_gamma <- read_csv("tables/simk3_tanh_gamma.csv")
#simk5_tanh_gamma <- read_csv("tables/simk5_tanh_gamma.csv")


# Function to create dataset for ggplot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df_boxplot <- function(mytable, m, Anames, models) {
  mytable <- mytable |> dplyr::select(-contains("pval"))
  colnames(mytable) <- c("dataset","model", Anames)
  #drop_datasets <- unique(mytable$dataset[is.na(mytable$A_01)])
  #mytable <- mytable[!(mytable$dataset %in% drop_datasets), ]
  mytable_long <- 
    mytable |> 
    pivot_wider(names_from = model,
                values_from = starts_with("A_"))
  
  if(ncol(mytable)==3) { # handling the k=2 case
    colnames(mytable_long) = c("dataset", 
                          paste0(A,"_True_diff"),
                          paste0(A, "_NN_est"),
                          paste0(A, "_LogitExpo_est"),
                          paste0(A,"_Naive_est"))
    }
  
  for(m in models) { 
    for(A in Anames) {
      mytable_long$temp <- 
        mytable_long[[paste0(A,"_True_diff")]] - mytable_long[[paste0(A,"_", m,"_est")]]
      colnames(mytable_long)[colnames(mytable_long) == "temp"] <- paste0(A, "_diff_",m)
    }
  }
  
  mytable_long <- 
    mytable_long |> 
    dplyr::select(dataset, contains("diff_")) |>
    pivot_longer(cols = starts_with("A_"),
                 names_to = "model",
                 values_to = "diffs") |>
    mutate(delta = str_sub(model, 1, 4),
           model = str_to_lower(str_sub(model, 11))) |>
    dplyr::select(dataset, delta, model, diffs) |>
    mutate(model = factor(model, 
                          levels = c("naive","logitexpo","nn"),
                          labels = c("naive", "logistic-expo", "NN")))
  
  
  mytable_long
}

# ~~~~~~~~ #
# Boxplot  #
# ~~~~~~~~ #

# params
flavor_ops <- c("tanh","gamma")
mytable <- simk3_tanh_gamma
k = 3
m <- combn(k, 2)-1
Anames <- paste0("A_", apply(m, 2, function(x) paste(x, collapse = "")))
models <- c("Naive","LogitExpo","NN")

mytable_long <- df_boxplot(mytable, m, Anames, models)
mytitle <- bquote(paste("True model is ", .(flavor_ops[[1]]), "-", .(flavor_ops[[2]])))

# all pooled
p1 <-
  ggplot2::ggplot(mytable_long, aes(x = model, y = diffs)) +
  geom_boxplot(fill = "skyblue") + 
  #ylim(c(-0.75,0.75)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.y = element_text(size = 12),
        axis.text = element_text(size = 12)) + 
  labs(title = mytitle, y = "diff", x="") -> p1

# by trt level
plist <- list()
i = 0
for(A in Anames){
  i <- i + 1
  plist[[i]] <- 
    ggplot2::ggplot(mytable_long[mytable_long$delta==A,], aes(x = model, y = diffs)) +
    geom_boxplot(fill = "skyblue") + 
    #ylim(c(-0.75,0.75)) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 16),
          axis.title.y = element_text(size = 12),
          axis.text = element_text(size = 12)) + 
    labs(title = paste0("For ", A), y = "diff", x="")
  plist[i]
}


# Saving plots
p1
ggsave(paste0("images/simk",k,"_",flavor_ops[1],flavor_ops[2],"_boxplot_pooled.jpeg"),
       width = 7.15, height = 4.95, dpi = 150)

combined_plot <- 
  wrap_plots(plist, ncol = k) +
  plot_annotation(title = mytitle)
combined_plot
ggsave(paste0("images/simk",k,"_",flavor_ops[1],flavor_ops[2],"_boxplot.jpeg"),
       width = 7.15, height = 4.95, dpi = 150)




#### DEPRECATED
#### Combine two sets of boxplots
### flavor_ops <- c("tanh","sigmoid", function(x) 1/(1+exp(-x)) * 10)
### mytable <- simk3_tanh_sigmoid
### mytable_long <- df_boxplot(mytable, m, Anames, models)
### mytitle <- bquote(paste("True model is ", .(flavor_ops[[1]]), "-", .(flavor_ops[[2]])))
### ggplot2::ggplot(mytable_long, aes(x = model, y = diffs)) +
###   geom_boxplot(fill = "skyblue") + 
###   ylim(c(-1,1.2)) +
###   theme_minimal() +
###   theme(plot.title = element_text(hjust = 0.5, size = 12),
###         axis.title.y = element_text(size = 12),
###         axis.text = element_text(size = 12)) + 
###   labs(title = mytitle, y = "diff", x="") -> p2
### 
### 
### combined_side <- (p1+p2) + 
###   plot_annotation(title = bquote(paste("Errors between ", Delta, " and ", hat(Delta), " per model for k=", .(k))),
###                   theme = theme(plot.title = element_text(size=16, hjust = 0.5)))
### 
### ggsave(paste0("images/simk",k,"_combined_side_by_side.jpeg"),
###        width = 7.15, height = 4.95, dpi = 150)

