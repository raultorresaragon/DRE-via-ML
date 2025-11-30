library(tidyverse)
library(ggplot2)
library(patchwork)

rm(list = ls())
flavor_ops <- c("logit","expo","ols",5)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Function to create dataset for ggplot #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

df_boxplot <- function(mytable, m, Anames, models) {
  mytable_nopval <- mytable |> dplyr::select(-contains("pval"))
  mytable_nopval$estimate <- stringr::str_to_lower(mytable_nopval$estimate)
  mytable_nopval$estimate <- stringr::str_replace_all(mytable_nopval$estimate, "logit", "l")
  colnames(mytable_nopval) <- c("dataset","model", Anames)
  #drop_datasets <- unique(mytable$dataset[is.na(mytable$A_01)])
  #mytable <- mytable[!(mytable$dataset %in% drop_datasets), ]
  mytable_long <- 
    mytable_nopval |> 
    pivot_wider(names_from = model,
                values_from = starts_with("A_"))
  
  if(ncol(mytable_nopval)==3) { # handling the k=2 case
    A <- Anames
    colnames(mytable_long) = c("dataset", 
                          paste0(A, paste0("_true_diff")),
                          paste0(A, paste0("_nn_est")),
                          paste0(A, paste0("_", models[2],"_est")),
                          paste0(A, paste0("_naive_est")))
  }
  
  for(m in models) { 
    for(A in Anames) {
      mytable_long$temp <- 
        mytable_long[[paste0(A,"_true_diff")]] - mytable_long[[paste0(A,"_", m,"_est")]]
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
                          levels = c('naive', models[2], "nn"),
                          labels = c('naive', models[2], "nn")))
  mytable_long
}

# ~~~~~~~~ #
# Boxplot  #
# ~~~~~~~~ #

# params

save_boxplots <- function(flavor_ops) {
  k <- flavor_ops[4]
  mytable <- readr::read_csv(paste0("_1trt_effect/tables/simk",
                                    flavor_ops[4],"_",
                                    flavor_ops[1],"_",
                                    flavor_ops[2],
                                    "_est_with_",
                                    flavor_ops[3],".csv"))
  m <- combn(as.numeric(k), 2)-1
  Anames <- paste0("A_", apply(m, 2, function(x) paste(x, collapse = "")))
  models <- c("naive",paste0('l', flavor_ops[3]),"nn")

  print(mytable)
  mytable_long <- df_boxplot(mytable, m, Anames, models)
  #mytitle <- bquote(paste("True model is ", .(flavor_ops[[1]]), "-", .(flavor_ops[[2]])))
  mytitle <- paste("True model is ", flavor_ops[1], "-", flavor_ops[2])
  print(mytable_long)

  # all pooled
  if(k<5){
  p1 <-
    ggplot2::ggplot(mytable_long, aes(x = model, y = diffs)) +
    geom_boxplot(fill = "skyblue") + 
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 16),
          axis.title.y = element_text(size = 12),
          axis.text = element_text(size = 12)) + 
    labs(title = mytitle, y = "diff", x="")
  
  
    ggsave(paste0("_1trt_effect/images/simk",k,"_",flavor_ops[1],flavor_ops[2],"_with_",flavor_ops[3],"_boxplot_pooled.jpeg"),
         width = 7.15, height = 4.95, dpi = 150)
  }

  # by trt level
  if(flavor_ops[4]>2) {
    plist <- list()
    i = 0
    for(A in Anames){
      i <- i + 1
      plist[[i]] <- 
       ggplot2::ggplot(mytable_long[mytable_long$delta==A,], aes(x = model, y = diffs)) +
        geom_boxplot(fill = "skyblue") + 
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5, size = 16),
              axis.title.y = element_text(size = 12),
              axis.text = element_text(size = 12)) + 
        labs(title = paste0("For ", A), y = "diff", x="")
      plist[i]
    }
    combined_plot <- 
      wrap_plots(plist, ncol = as.numeric(k)) +
      plot_annotation(title = mytitle)
    combined_plot
    ggsave(paste0("_1trt_effect/images/simk",k,"_",flavor_ops[1],flavor_ops[2],"_with_",flavor_ops[3],"_boxplot.jpeg"),
           width = 7.15, height = 4.95, dpi = 150)
  }
}

pflavs <- c("l","t")
oflavs <- c("e","s","l","g")
flavors <- #pairwise combination of flavors
  tidyr::expand_grid(pflavs, oflavs) |> 
  dplyr::mutate(l = paste0(pflavs, oflavs)) |> 
  dplyr::pull("l")  
if(length(flavors)==8) flavors <- flavors[c(1,6,7,8)]

for(k in c(5)){
  for(mod in c("expo","ols")) {
    for(flav in flavors) {
    if(flav == "le") flavor_ops <- c("logit","expo", mod, k) 
    if(flav == "ls") flavor_ops <- c("logit","sigmoid", mod, k) 
    if(flav == "ll") flavor_ops <- c("logit","lognormal", mod, k) 
    if(flav == "lg") flavor_ops <- c("logit","gamma", mod, k) 
    
    if(flav == "te") flavor_ops <- c("tanh", "expo", mod, k) 
    if(flav == "ts") flavor_ops <- c("tanh", "sigmoid", mod, k) 
    if(flav == "tl") flavor_ops <- c("tanh", "lognormal", mod, k) 
    if(flav == "tg") flavor_ops <- c("tanh", "gamma", mod, k) 
    save_boxplots(flavor_ops)
    }
  }
}



flavor_ops <- c("logit","expo","expo",3)
flavor_ops <- c("logit","expo","ols",2)
flavor_ops <- c("tanh","sigmoid","expo",2)
flavor_ops <- c("tanh","gamma","expo",2)
flavor_ops <- c("tanh","lognormal","expo",2)
flavor_ops <- c("tanh","sigmoid","ols",2)
flavor_ops <- c("tanh","gamma","ols",2)
flavor_ops <- c("tanh","lognormal","ols",2)














### simk2_logit_expo <- read_csv("tables/simk2_logit_expo.csv")
### simk3_logit_expo <- read_csv("tables/simk3_logit_expo.csv")
### simk5_logit_expo <- read_csv("tables/simk5_logit_expo.csv")
### 
### simk2_tanh_sigmoid <- read_csv("tables/simk2_tanh_sigmoid.csv")
### simk3_tanh_sigmoid <- read_csv("tables/simk3_tanh_sigmoid.csv")
### simk5_tanh_sigmoid <- read_csv("tables/simk5_tanh_sigmoid.csv")
### 
### simk2_tanh_lognormal <- read_csv("tables/simk2_tanh_lognormal.csv")
### simk3_tanh_lognormal <- read_csv("tables/simk3_tanh_lognormal.csv")
### simk5_tanh_lognormal <- read_csv("tables/simk5_tanh_lognormal.csv")
### 
### simk2_tanh_gamma <- read_csv("tables/simk2_tanh_gamma.csv")
### simk3_tanh_gamma <- read_csv("tables/simk3_tanh_gamma.csv")
### simk5_tanh_gamma <- read_csv("tables/simk5_tanh_gamma.csv")
###
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

