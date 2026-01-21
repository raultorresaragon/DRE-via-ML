# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Raul
# File name: real_data_analysis_00.R
# Date: 2025-07-08
# Note: This script imports and inspects
#       a data set provided by Dr. Ahn
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

library(tidyverse)
library(readxl)
rm(list = ls())

df <- read_xlsx("real_data/ASTR_dataset.xlsx")
colnames(df) <- c("id", "gender", "age", "time", "gh", "chemo", "partial_or_total_removal")

# N
dim(df)[1]

# number of participants
length(unique(df$id))

# number of measurements per participant
count(df, id) |> pull("n") |> table()

# -1 means missing
recode_NA <- function(df, x){
  df[[x]][df[[x]]==-1] <- NA
  df
}
for(x in colnames(df)) {
  df <- recode_NA(df, x)
}

recode_01 <- function(df, x, value1, value0=0, char=FALSE) {
  if(char==TRUE) { value1 <- char(value1); value2 <- char(value2)}
  df[[x]][df[[x]]==value0] <- 0
  df[[x]][df[[x]]==value1] <- 1
  df
}

# recode binary variables
count(df, gender)
  df <- recode_01(df, "gender", value1="F", value0="M")
  
count(df, chemo)
  df <- recode_01(df, "chemo", value1=2, value0=1)

count(df, partial_or_total_removal)
  df <- recode_01(df, "partial_or_total_removal", value1=2, value0=1)
  

# Keep only the first measurement since we're doing cross-section for now
df_t1 <- df |> filter(time==1)
  
  
# tabulate categorical and binary varibles
count(df_t1, gender)
count(df_t1, chemo)
count(df_t1, partial_or_total_removal)
  
# summarize continuous variables
summary(df_t1$gh)
hist(df_t1$gh)

# output file
write_csv(df, "real_data/recoded_ASTR.csv")
write_csv(df_t1, "real_data/recoded_ASTR_t1.csv")
