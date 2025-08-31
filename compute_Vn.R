# ----------
# Compute Vn
# ----------
get_Vn <- function(fit_Y_nn, X_new) {
  V_n <- tibble(V_ = rep(NA, nrow(X_new)))
  for(A_type in names(fit_Y_nn)) {
    for(j in c(2,3)) {
      V_n <- 
        V_n |>
        mutate(V_ = predict(fit_Y_nn[[A_type]][[j]], new_data = X_new, type = "raw") |>
                 as.vector())
      
      V_type <- stringr::str_replace(A_type, "A", "V")
      s <- ifelse(j==2, j+2, j)
      r <- stringr::str_sub(A_type, s, s)
      colnames(V_n)[colnames(V_n) == "V_"] <- paste0(V_type, "_g", r)
    }
  }
  V_n |> mutate(OTR = stringr::str_sub(colnames(V_n)[max.col(V_n)], 6, 7))
}
