library(tidyverse)
library(ggplot2)
library(patchwork)

rm(list = ls())
simk3_logit_expo <- read_csv("tables/simk3_logit_expo.csv")
simk3_tanh_sigmoid <- read_csv("tables/simk3_tanh_sigmoid.csv")
simk5_logit_expo <- read_csv("tables/simk5_logit_expo.csv")
simk5_tanh_sigmoid <- read_csv("tables/simk5_tanh_sigmoid.csv")


df_boxplot <- function(mytable, m, Anames, models) {
  colnames(mytable) <- c("dataset","model", Anames)
  mytable_long <- 
    mytable |> 
    pivot_wider(names_from = model,
                values_from = starts_with("A_"))
  
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
# K=3 plot #
# ~~~~~~~~ #

# params
k = 3
m <- combn(k, 2)-1
Anames <- paste0("A_", apply(m, 2, function(x) paste(x, collapse = "")))
models <- c("Naive","LogitExpo","NN")
flavor_ops <- c("logit","expo", exp)
mytable <- simk3_logit_expo


flavor_ops <- c("logit","expo", exp)
mytable <- simk3_logit_expo
mytable_long <- df_boxplot(mytable, m, Anames, models)
mytitle <- bquote(paste("True model is ", .(flavor_ops[[1]]), "-", .(flavor_ops[[2]])))
ggplot2::ggplot(mytable_long, aes(x = model, y = diffs)) +
  geom_boxplot(fill = "skyblue") + 
  ylim(c(-1,1.2)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 12),
        axis.title.y = element_text(size = 12),
        axis.text = element_text(size = 12)) + 
  labs(title = mytitle, y = "diff", x="") -> p1

flavor_ops <- c("tanh","sigmoid", function(x) 1/(1+exp(-x)) * 10)
mytable <- simk3_tanh_sigmoid
mytable_long <- df_boxplot(mytable, m, Anames, models)
mytitle <- bquote(paste("True model is ", .(flavor_ops[[1]]), "-", .(flavor_ops[[2]])))
ggplot2::ggplot(mytable_long, aes(x = model, y = diffs)) +
  geom_boxplot(fill = "skyblue") + 
  ylim(c(-1,1.2)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 12),
        axis.title.y = element_text(size = 12),
        axis.text = element_text(size = 12)) + 
  labs(title = mytitle, y = "diff", x="") -> p2


combined_side <- (p1+p2) + 
  plot_annotation(title = bquote(paste("Errors between ", Delta, " and ", hat(Delta), " per model for k=", .(k))),
                  theme = theme(plot.title = element_text(size=16, hjust = 0.5)))

ggsave(paste0("images/simk",k,"_combined_side_by_side.jpeg"),
       width = 7.15, height = 4.95, dpi = 150)


# ~~~~~~~~ #
# K=5 plot #
# ~~~~~~~~ #

# params
k = 5
m <- combn(k, 2)-1
Anames <- paste0("A_", apply(m, 2, function(x) paste(x, collapse = "")))
models <- c("Naive","LogitExpo","NN")
flavor_ops <- c("logit","expo", exp)

flavor_ops <- c("logit","expo", exp)
mytable <- simk5_logit_expo
mytable_long <- df_boxplot(mytable, m, Anames, models)
mytitle <- bquote(paste("True model is ", .(flavor_ops[[1]]), "-", .(flavor_ops[[2]])))
ggplot2::ggplot(mytable_long, aes(x = model, y = diffs)) +
  geom_boxplot(fill = "skyblue") + 
  ylim(c(-1,1.2)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 12),
        axis.title.y = element_text(size = 12),
        axis.text = element_text(size = 12)) + 
  labs(title = mytitle, y = "diff", x="") -> p1

flavor_ops <- c("tanh","sigmoid", function(x) 1/(1+exp(-x)) * 10)
mytable <- simk5_tanh_sigmoid
mytable_long <- df_boxplot(mytable, m, Anames, models)
mytitle <- bquote(paste("True model is ", .(flavor_ops[[1]]), "-", .(flavor_ops[[2]])))
ggplot2::ggplot(mytable_long, aes(x = model, y = diffs)) +
  geom_boxplot(fill = "skyblue") + 
  ylim(c(-1,1.2)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 12),
        axis.title.y = element_text(size = 12),
        axis.text = element_text(size = 12)) + 
  labs(title = mytitle, y = "diff", x="") -> p2


combined_side <- (p1+p2) + 
  plot_annotation(title = bquote(paste("Errors between ", Delta, " and ", hat(Delta), " per model for k=", .(k))),
                  theme = theme(plot.title = element_text(size=16, hjust = 0.5)))

ggsave(paste0("images/simk",k,"_combined_side_by_side.jpeg"),
       width = 7.15, height = 4.95, dpi = 150)



# DEPRECATED CODE

### colnames(mytable) <- c("dataset","model","A_01","A_02","A_12")
### mytable_long <- 
###     mytable |> 
###     pivot_wider(names_from = model,
###                 values_from = starts_with("A_")) |>
###     mutate(A_01_diff_naive =     A_01_True_diff - A_01_Naive_est,
###            A_01_diff_logitexpo = A_01_True_diff - A_01_LogitExpo_est,
###            A_01_diff_nn =        A_01_True_diff - A_02_NN_est,
###            A_02_diff_naive =     A_02_True_diff - A_02_Naive_est,
###            A_02_diff_logitexpo = A_02_True_diff - A_02_LogitExpo_est,
###            A_02_diff_nn =        A_02_True_diff - A_01_NN_est,
###            A_12_diff_naive =     A_12_True_diff - A_12_Naive_est,
###            A_12_diff_logitexpo = A_12_True_diff - A_12_LogitExpo_est,
###            A_12_diff_nn =        A_12_True_diff - A_12_NN_est
###            ) |>
###     dplyr::select(dataset, contains("diff_")) |>
###     pivot_longer(cols = starts_with("A_0"),
###                  names_to = "model",
###                  values_to = "diffs") |>
###     mutate(delta = str_sub(model, 1, 4),
###            model = str_sub(model, 11)) |>
###     dplyr::select(dataset, delta, model, diffs) |>
###     mutate(model = factor(model, 
###                           levels = c("naive","logitexpo","nn"),
###                           labels = c("naive", "logistic-expo", "NN")))
### 
### mytitle <- bquote(paste("True model is ", .(flavor_ops[[1]]), "-", .(flavor_ops[[2]])))
### ggplot2::ggplot(mytable_long, aes(x = model, y = diffs)) +
###   geom_boxplot(fill = "skyblue") + 
###   ylim(c(-1,1.2)) +
###   theme_minimal() +
###   theme(plot.title = element_text(hjust = 0.5, size = 12),
###         axis.title.y = element_text(size = 12),
###         axis.text = element_text(size = 12)) + 
###   labs(title = mytitle, y = "diff", x="") -> p1
### 
### 
### 
### 
### flavor_ops <- c("tanh","sigmoid", function(x) 1/(1+exp(-x)) * 10)
### mytable <- simk3_tanh_sigmoid
### colnames(mytable) <- c("dataset","model","A_01","A_02","A_12")
### mytable_long <- 
###   mytable |> 
###   pivot_wider(names_from = model,
###               values_from = starts_with("A_")) |>
###   mutate(A_01_diff_naive =     A_01_True_diff - A_01_Naive_est,
###          A_01_diff_logitexpo = A_01_True_diff - A_01_LogitExpo_est,
###          A_01_diff_nn =        A_01_True_diff - A_02_NN_est,
###          A_02_diff_naive =     A_02_True_diff - A_02_Naive_est,
###          A_02_diff_logitexpo = A_02_True_diff - A_02_LogitExpo_est,
###          A_02_diff_nn =        A_02_True_diff - A_01_NN_est,
###          A_12_diff_naive =     A_12_True_diff - A_12_Naive_est,
###          A_12_diff_logitexpo = A_12_True_diff - A_12_LogitExpo_est,
###          A_12_diff_nn =        A_12_True_diff - A_12_NN_est
###   ) |>
###   dplyr::select(dataset, contains("diff_")) |>
###   pivot_longer(cols = starts_with("A_0"),
###                names_to = "model",
###                values_to = "diffs") |>
###   mutate(delta = str_sub(model, 1, 4),
###          model = str_sub(model, 11)) |>
###   dplyr::select(dataset, delta, model, diffs) |>
###   mutate(model = factor(model, 
###                         levels = c("naive","logitexpo","nn"),
###                         labels = c("naive", "logistic-expo", "NN")))
### 
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
###   plot_annotation(title = bquote(paste("Errors between ", Delta, " and ", hat(Delta), " per model.")),
###                   theme = theme(plot.title = element_text(size=16, hjust = 0.5)))
### 
### ggsave(paste0("images/simk",k,"_combined_side_by_side.jpeg"),
###        width = 7.15, height = 4.95, dpi = 150)

