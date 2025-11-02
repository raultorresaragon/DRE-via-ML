# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: one_sim_k3plus.R
# Date: 2025-05-15
# Note: This script creates a function
#       to run one iteration of k3 plus sims
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ---------------------------
# PARAMERTERS FOR DEBUGGING

    rm(list = ls())
    set.seed(1609) #set.seed(1810) #set.seed(505)
    M <- 1
    k <- 3
    zero_effect = FALSE
    if(k==2){ p<-3}
    if(k==3){ p<-8}
    if(k==5){ p<-12}
    n <- k*200
    A_flavor = "logit"
    Y_flavor = "expo"
    beta_Y_scalar = 1
    gamma <- c(0.8, 0.6, 0.52, 0.37)[1:(k-1)] * as.numeric(!zero_effect)
    rho   <- round(runif(1, 0.4, 0.6),1)
    Xmu   <- round(runif(p, -1, 1),1)
    beta_A <- matrix(rep(1,(k-1)), nrow=1) |> 
                rbind(matrix(round(runif(p*(k-1), -2, 2),1), nrow=p))
    beta_Y <- c(1, round(runif(p, -1, 1),1)) * beta_Y_scalar
    hidunits = c(2,6)
    eps = c(120,150)
    penals = c(0.001,0.005)
    
    iter = 1; 
    source("YAX_funs.R")
    source("outcome_models.R")
    source("pscores_models.R")
    source("get_diff.R")
    source("get_true_diff.R")
    source("compute_Vn.R")
    source("Y_Yhat_sorted_plots.R")
    verbose=FALSE
    export_images = FALSE
    root = paste0(getwd(),"/_", as.numeric(!zero_effect), "trt_effect/")

# ---------------------------

one_sim <- function(n, p, Xmu, beta_A, beta_Y, gamma, k,
                    A_flavor, Y_flavor, 
                    Y_param = "expo",
                    hidunits, eps, penals, verbose = FALSE, iter = 1, 
                    export_images = FALSE) {
  
  X <- gen_X(n=n, p=p, rho=rho, mu=Xmu, p_bin = 1) # floor(p * 1/3))
  A <- gen_A(X=X, beta=beta_A, flavor_A=A_flavor)
  Y <- gen_Y(X=X, A=A, beta_Y=beta_Y, gamma=gamma, flavor_Y=Y_flavor)$Y
  dat <- cbind(Y,A,X) 
  stopifnot(Y>=0)
  readr::write_csv(dat, 
                   paste0(root,"datasets/df_k", k, A_flavor,Y_flavor,"_dset",iter,".csv"))
  
  # plot Y
  main <- paste0("k=",k,"  flavor:", A_flavor, "-", Y_flavor, "\nN=",n, "  dim(X)=", p)
  xb_Y <-(as.matrix(cbind(1,X))%*%beta_Y) 
  mycols <- c("black","darkred","green","blue","skyblue")
  par(mfrow=c(1,1)) 
  plot(Y~xb_Y, main=main, cex.main=2, col=as.factor(A))
  legend("topright", legend = paste0("A=", 0:(k-1)), col=mycols[1:k], pch=1)
  par(mfrow=c(1,1))
  jpeg(paste0(root,"images/genYplots/genY_", k, A_flavor, Y_flavor, "_dset", iter, ".jpeg"), 
       width = 1000, height = 510)
      par(mfrow=c(1,2)) 
      plot(Y~xb_Y, main=main, cex.main=2, col=as.factor(A))
      legend("topright", legend = paste0("A=", 0:(k-1)), col=mycols[1:k], pch=1)
      i=0
      plot(sort(Y[A==i]), main="sorted", cex.main=2, col=mycols[i+1], 
           ylim=c(min(Y), max(Y)), xlim=c(0, n))
      for(i in 0:(k-1)){
        points(sort(Y[A==i]), main=main, cex.main=2, col=mycols[i+1])
      }
      legend("topright", legend = paste0("A=", 0:(k-1)), col=mycols[1:k], pch=1)
  dev.off()
  par(mfrow=c(1,1))
  
  # print P(A=j)
  for(i in 1:k-1) {cat(paste0("\n  P(A=",i,")= ", mean(A==i) |> round(1)))}
  
  # print delta_ij
  Y_fun <- gen_Y(X=X, A=A, beta_Y=beta_Y, gamma=gamma, flavor_Y=Y_flavor)$fun_Y
  true_diffs <- 
    apply(as.matrix(combn(k,2)-1), 
          2, 
          function(x) get_true_diff(x, 
                                    xb_Y=xb_Y, 
                                    gamma=gamma, 
                                    Y_flavor=Y_flavor))
  rm(xb_Y)
  
  # Estimate A (propensity model)
  tictoc::tic("\nA model")
  fit_A_nn <- estimate_A_nn(X=X, dat=dat, k=k, 
                            hidunits=hidunits,
                            eps=eps, 
                            penals=penals,
                            verbose=verbose)
  fit_A_logit <- estimate_A_logit(X=X, dat=dat, k=k, verbose=verbose)
  tictoc::toc()  
  
  # Estimate Y (outcome model)
  tictoc::tic("\nY model")
  fit_Y_nn <- estimate_Y_nn(dat, pscores_df=fit_A_nn$pscores, k=k,
                            hidunits=hidunits,
                            eps=eps, 
                            penals=penals, 
                            verbose=verbose)
  if(Y_param == "expo") {
    fit_Y_param <- estimate_Y_expo(dat, pscores_df=fit_A_logit$pscores, k=k)
  } else {
    fit_Y_param <- estimate_Y_ols(dat, pscores_df=fit_A_logit$pscores, k=k)
  }
  tictoc::toc()
  
  # Save predicted Aj and Yj plot
  plot_predicted_A_Y(beta_A, beta_Y, dat, 
                     fit_Y_nn, fit_Y_param, gamma, 
                     fit_A_nn, fit_A_logit, A_flavor, Y_flavor, ds=iter, k, 
                     save=export_images)
  
  # Pack results into k rows
  get_naive_est <- function(x) {
    i <- x[[2]]
    j <- x[[1]]
    d <- mean(Y[A==i]) - mean(Y[A==j])
    cat(paste0("\n  Naive diff means_{", j,",", i, "} = ", round(d, 3)))
    d
  }
  naive_est <- apply(as.matrix(combn(k,2) - 1), 2, get_naive_est)
  nn_model_est <- sapply(fit_Y_nn, function(x) {x[[1]]}) |> data.frame() 
  logit_expo_est <- sapply(fit_Y_param, function(x) {x[[1]]}) |> data.frame()
  
  nn_model_pvals <- nn_model_est[5,]
  logit_expo_pvals <- logit_expo_est[5,]
  
  
  # Delta estimates
  my_k_rows <- 
    nn_model_est[1,] |> 
    rbind(logit_expo_est[1,]) |> 
    rbind(naive_est, true_diffs) |> 
    `row.names<-`(c("NN_est", paste0("Logit",Y_param,"_est"), "Naive_est","True_diff")) |>
    as.data.frame() |>
    rownames_to_column("estimate") |>
    mutate(dataset = iter) |> 
    dplyr::select(dataset, everything())
  
  # Pvalues
  my_k_rows_pvals <- 
    nn_model_est[5,] |> 
    rbind(logit_expo_est[5,]) |> 
    rbind(naive_est*NA, true_diffs*NA) |> 
    `row.names<-`(c("NN_est", paste0("Logit",Y_param,"_est"), "Naive_est","True_diff")) |>
    as.data.frame() |>
    rownames_to_column("estimate") |>
    mutate(dataset = iter) |> 
    dplyr::select(dataset, everything()) 
  colnames(my_k_rows_pvals)[3:ncol(my_k_rows_pvals)] <-
    c(paste0(colnames(my_k_rows_pvals)[3:ncol(my_k_rows_pvals)], "_pval"))
  
  my_k_rows <- inner_join(my_k_rows, my_k_rows_pvals, 
                          by = c("dataset" = "dataset", "estimate" = "estimate"))

  my_k_rows <- rbind(my_k_rows[4,], my_k_rows[-4,]) |> `row.names<-.data.frame`(1:4)
  
  
  # Compute Vn
  X_new <- matrix(round(runif(5*dim(X)[2], -8, 8),1), 
                  nrow = 5, 
                  byrow = TRUE) |> as.data.frame()
  colnames(X_new) <- colnames(X)
  Vn_df <- 
    get_Vn(fit_Y_nn, X_new = X_new) |> 
    mutate(dataset = iter) |> 
    dplyr::select(dataset, everything())
  
  list(my_k_rows = my_k_rows, Vn_df = Vn_df, Xnew_Vn = X_new)
  
}


