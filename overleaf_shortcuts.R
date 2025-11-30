# Overleaf shortcuts

\FloatBarrier
% Example 3: logit-expo
% Example 3: (a) tanh-sigmoid
% Example 3: (b) tanh-lognormal
% Example 3: (c) tanh-gamma
% -------------------------
  \noindent{\bf Example 2(b): tanh-lognormal}  

# TABLE
mytable <- 
  readr::read_csv("_0trt_effect/tables/simk3_logit_expo.csv") |>
  dplyr::select(!ends_with("pval"))
  
myOTR <- 
  readr::read_csv("_0trt_effect/tables/OTR_simk2_logit_expo.csv")
  #View(myOTR)

print(xtable::xtable(mytable, include.rownames = FALSE))


\begin{landscape}
\begin{table}[ht]
\centering
\footnotesize
\caption{Model estimates compared to true $\Delta_{i,j}$ from 5 simulated datasets with $k=5$ and $n=1200$. Data generating process is logit for $A_j$ and exponential for $Y_i$}
\vspace{1.5em} 
\begin{tabular}{cccccccccccc}
\hline
dataset & estimate & $\Delta_{0,1}$ & $\Delta_{0,2}$ & $\Delta_{0,3}$ & $\Delta_{0,4}$ & $\Delta_{1,2}$ & $\Delta_{1,3}$ & $\Delta_{1,4}$ & $\Delta_{2,3}$ & $\Delta_{2,4}$ & $\Delta_{3,4}$ \\ 
\hline

%paste results here

\hline
\end{tabular}
\label{tab:sim_k5_tanh_sigmoid}
\end{table}
\end{landscape}


# BOX PLOT
\begin{figure}[h]
\caption{Difference between $\hat{\Delta}_{j,j'}$ and true $\Delta_{j,j'}$ across 10 datasets by model.}
\center
\includegraphics[scale=0.4]{images/Example3/simk5_tanhsigmoid_boxplot.jpeg}
\label{fig:simk5_tanhsigmoid_boxplot}
\end{figure}



# CONCOMITANT PLOT
% FIGURE: 
% Concomitant   ~~~ 
\begin{figure}[h]
\caption{True and predicted $P(A_i)$ and observed and predicted $Y_i$ for $k=5$ when the data-generating process is logit-expo by model.}
\center
\includegraphics[scale=0.4]{images/Example3/YYhat_sorted_k5logitexpo.jpeg}
\label{fig:YYhat_sorted_k5logitexpo}
\end{figure}  



# OTR
\noindent{\textbf{Optimal treatment regime}}. To compute the optimal treatment 
regime $d^*(\bx)$, we first create a new observation 
$\bx'_i\in R^p$ where its values are draws from $\text{Unif}(-8,8)$. 
Obtaining covariate pattern $\bx'=\{-2.4,  -5, 7.3,  -2.5, 1.5, -5.8, 3.7, 3, 7.7, -0.2, 4.3, 1\}$, we find that the optimal treatment is $d^*(\bx)=1$. 

The OTR is $d^*(\bx)=2$ when covariate pattern is $\bx''=\{3.3 -5.1, -7.4, -4.2, 8.0, 0.4,-6.3,-6.2,3.0,6.6,8.0, 0\}$. 

But when covariate pattern changes to $\bx''=\{4.6, 1.2,  -5.3,  -2.3, 5, 3.3, 5.5, -4.8,   7.3, 6.3, -5.2, 1\}$, the OTR is then $d^*(\bx'')=2$. \\