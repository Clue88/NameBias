library(arrow)
library(fixest)
library(dplyr)
setwd("/Users/parasu/projects/NameBias")

df = read_parquet("data/reg_data.parquet", as_tibble=TRUE)


# process data
df$no_fixed_effects = 1

relevel_df <- function(data) {
  data <- within(data, race <- relevel(factor(race), ref = "nh_white"))
  data <- within(data, sex <- relevel(factor(sex), ref = "M"))
  return(data)
}
df = relevel_df(df)
df$log_pile_freq = log(df$frequency_pile + 1) 
df$log_fl_freq = log(df$frequency_FL_corpus + 1)


m.base = feols(mean_mean_pool_sim_score ~ race + sex | no_fixed_effects, weights=df$count, data = df)
m.full = feols(mean_mean_pool_sim_score ~ race + sex + n_tokens + log_pile_freq + log_fl_freq | no_fixed_effects, weights=df$count, data = df)
m.int = feols(mean_mean_pool_sim_score ~ race*sex + n_tokens + log_pile_freq + log_fl_freq | no_fixed_effects, weights=df$count, data = df)

etable(m.base, m.full, m.int)


m.base_cls = feols(mean_cls_sim_score ~ race + sex | no_fixed_effects, weights=df$count, data = df)
m.full_cls = feols(mean_cls_sim_score ~ race + sex + n_tokens + log_pile_freq + log_fl_freq | no_fixed_effects, weights=df$count, data = df)
m.int_cls = feols(mean_cls_sim_score ~ race*sex + n_tokens + log_pile_freq + log_fl_freq | no_fixed_effects, weights=df$count, data = df)
etable(m.base_cls, m.full_cls, m.int_cls)
