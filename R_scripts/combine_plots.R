library(tidyverse)
library(magick)

#### Read the images
###img1 <- image_read("images/simk2_logit_expo.jpeg")
###img2 <- image_read("images/simk2_tanh_sigmoid.jpeg")
###
#### Resize to same height if needed (optional but recommended)
###img2 <- image_resize(img2, geometry = geometry_size_percent(width = NULL, height = 100))
###img1 <- image_resize(img1, geometry = geometry_size_percent(width = NULL, height = 100))
###
#### Combine images side by side
###combined <- image_append(c(img1, img2),stack = TRUE)
###
#### Save combined image
###image_write(combined, path = "images/simk2_boxplots_combined.jpeg", format = "jpg")




flavor_ops <- c("logit","expo", exp)
mytable <- simk2_logit_expo
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
                            labels = c("naive", "logistic-expo", "NN")))

mytitle <- bquote(paste("True model is ", .(flavor_ops[[1]]), "-", .(flavor_ops[[2]])))
ggplot2::ggplot(mytable_long, aes(x = estimator, y = diffs)) +
  geom_boxplot(fill = "skyblue") + 
  ylim(c(-1,1.2)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 12),
        axis.title.y = element_text(size = 12),
        axis.text = element_text(size = 12)) + 
  labs(title = mytitle, y = "diff", x="") -> p1


flavor_ops <- c("tanh","sigmoid", function(x) 1/(1+exp(-x)) * 10)
mytable <- simk2_tanh_sigmoid
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
                            labels = c("naive", "logistic-expo", "NN")))

mytitle <- bquote(paste("True model is ", .(flavor_ops[[1]]), "-", .(flavor_ops[[2]])))
ggplot2::ggplot(mytable_long, aes(x = estimator, y = diffs)) +
  geom_boxplot(fill = "skyblue") + 
  ylim(c(-1,1.2)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 12),
        axis.title.y = element_text(size = 12),
        axis.text = element_text(size = 12)) + 
  labs(title = mytitle, y = "diff", x="") -> p2


combined_side <- (p1+p2) + 
  plot_annotation(title = bquote(paste("Errors between ", Delta, " and ", hat(Delta), " per model.")),
                  theme = theme(plot.title = element_text(size=16, hjust = 0.5)))

ggsave(paste0("images/simk2_combined_side_by_side.jpeg"),
       width = 7.15, height = 4.95, dpi = 150)

ggsave(paste0("images/simk2_combined_side_by_side.jpeg"),
       width = 7.15*0.8, height = 4.95*0.8, dpi = 150)
