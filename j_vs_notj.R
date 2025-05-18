# computes muhat_i vs muhat_noti
### for(i in 1:k-1) {
###   
###   pi_hat_i <- pscores_df[,i+1] |> as.vector()
###   A_i <- if_else(dat$A==i, 1, 0)
###   delta_i <- as.numeric(A_i==1)
###   delta_0 <- as.numeric(A_i==0)
###   
###   g_i <- Y_model_nn(dat=dat[A_i==1,] |> dplyr::select(-A), y_func = "Y~.", 
###                     hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
###   ghat_i <- predict(g_i, new_data = dat %>% select(-Y, -A), type = "raw") |> as.vector()
###   
###   g_0 <- Y_model_nn(dat=dat[A_i==0, ] |> dplyr::select(-A), y_func = "Y~.",
###                     hidunits=hidunits, eps=eps, penals=penals, verbose=verbose)
###   ghat_0 <- predict(g_0, new_data = dat %>% select(-Y, -A), type = "raw") |> as.vector()
###   
###   d <- get_diff(ghat_i, delta_i, ghat_0, delta_0, pi_hat_i, Y)
###   
###   muhat_i <- d$muhat_1
###   
###   cat(paste0("\n  NN est diff means [k=",i, " vs. k=~",i,"]=", round(d$diff_means, 3)))
### 
###   o_i<- list(muhat_i = muhat_i, g_i = g_i, diff_i = d$diff_means)
###   
###   names(o_i) <- c(paste0("muhat_", i), paste0("g_", i), paste0("diff_", i))
###   
###   o[[i+1]] <- o_i
### }
### o