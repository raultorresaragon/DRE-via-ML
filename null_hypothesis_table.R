# null hypothesis table

k <- 2
amodel <- 'logit'
ymodel <- 'expo'

# TABLE

read_etable <- function(k,amodel,ymodel,pickdataset=1) {
  readr::read_csv(paste0("_0trt_effect/tables/simk",k,"_",amodel,"_",ymodel,".csv")) |>
  dplyr::mutate(across(ends_with("pval"), ~ 1 - .x)) |>
  dplyr::filter(dataset == pickdataset) |>
  dplyr::filter(estimate != "Naive_est" & estimate != "True_diff") |>
  dplyr::mutate(DGP = paste0(amodel,"_",ymodel)) |>
  dplyr::select(-dataset) |>
  dplyr::select(DGP, estimate, ends_with('pval')) |>
  dplyr::mutate(estimate = stringr::str_remove_all(estimate, '_est')) |>
  rename_with(~ gsub("^A_(\\d+)_pval$", "hat{Delta}_{\\1} pval", .x))
}

k2_zero_table <-
  read_etable(2, "logit", "expo", 8) |>
  rbind(read_etable(2,"tanh","sigmoid",5)) |>
  rbind(read_etable(2,"tanh","lognormal",2)) |>
  rbind(read_etable(2,"tanh","gamma",7))

k3_zero_table <-
  read_etable(3, "logit", "expo",6) |>
  rbind(read_etable(3,"tanh","sigmoid",6)) |>
  rbind(read_etable(3,"tanh","lognormal",9)) |>
  rbind(read_etable(3,"tanh","gamma",6))

k5_zero_table <-
  read_etable(5, "logit", "expo",4) |>
  rbind(read_etable(5,"tanh","sigmoid",2)) |>
  rbind(read_etable(5,"tanh","lognormal",4)) |>
  rbind(read_etable(5,"tanh","gamma",6))

print(xtable::xtable(k5_zero_table, include.rownames = FALSE))

