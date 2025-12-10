###############################################################################
#                         ML BASED MODEL PIPELINE
# -----------------------------------------------------------------------------
# Description:
# End-to-end Machine Learning workflow for HYQVIA PI Model:
#  - Data transformation (Adstock + Power + Lag)
#  - Feature engineering and aggregation
#  - Random Forest model training
#  - SHAP-based contribution decomposition
#  - Full SHAP Values in Long Format
#  - Model evaluation (R², MAPE)
#  - Permutation-based variable importance
#  - Response curve generation (Concave / Hill)
###############################################################################


# =========================================================================== #
#  1. SETUP & LIBRARIES
# =========================================================================== #
rm(list = ls())
gc()

library(dplyr)
library(tidyr)
library(data.table)
library(ggplot2)
library(randomForest)
library(fastshap)
library(iml)
library(caret)
library(car)

# --- Utility: Copy DataFrame to Clipboard ---
copy_clipboard <- function(db) {
  write.table(db, "clipboard-16384", sep = "\t", col.names = TRUE, row.names = FALSE)
}


# =========================================================================== #
#  2. TRANSFORMATION FUNCTION (Adstock + Power + Lag)
# =========================================================================== #
transformation_function <- function(data, variable_list, adstock_rate, FR_rate, lag_rate) {
  setDT(data)
  
  for (i in seq_along(variable_list)) {
    var <- variable_list[i]
    adstock <- adstock_rate[i]
    fr <- FR_rate[i]
    lag <- lag_rate[i]
    
    # ---- Adstock transformation ----
    data[, paste0(var, "_transform") := {
      actual_values <- as.numeric(get(var))
      transformed <- numeric(.N)
      for (idx in seq_along(actual_values)) {
        if (idx == 1) {
          transformed[idx] <- actual_values[idx]
        } else {
          adstocked_sum <- actual_values[idx]
          for (j in 1:(idx - 1)) {
            adstocked_sum <- adstocked_sum + actual_values[idx - j] * (adstock ^ j)
          }
          transformed[idx] <- adstocked_sum
        }
      }
      transformed
    }, by = Trans_Level]
    
    # ---- Power transformation ----
    data[, paste0(var, "_transform_fr") := get(paste0(var, "_transform")) ^ fr]
    
    # ---- Lag transformation ----
    if (!is.na(lag) && lag > 0) {
      data[, paste0(var, "_final") := shift(get(paste0(var, "_transform_fr")), lag, fill = NA), by = Trans_Level]
    } else {
      data[, paste0(var, "_final") := get(paste0(var, "_transform_fr"))]
    }
    
    # ---- Rename final variable ----
    new_name <- paste(var, "ad", adstock, "fr", fr, "lag", lag, sep = "_")
    setnames(data, old = paste0(var, "_final"), new = new_name)
    
    # ---- Clean intermediate variables ----
    data[, c(paste0(var, "_transform"), paste0(var, "_transform_fr")) := NULL]
  }
  
  return(data)
}


# =========================================================================== #
#  3. LOAD & PREPARE DATA
# =========================================================================== #
setwd("C:\\Users\\Rohit.Wagh\\OneDrive - EVERSANA\\Desktop\\MMx_ML\\Takeda\\HY_PI")
raw_data <- read.csv("HYQVIA_PI_Model_data_19_06_2025.csv", stringsAsFactors = FALSE)

head(raw_data)

# View the column names of the dataset
names(raw_data)

raw_data$Date <- as.Date(raw_data$Date, "%m/%d/%Y")

unique(raw_data$Date)
# Define key variables
raw_data$Trans_Level <- raw_data$DMA
raw_data$modeling_Var <- raw_data$DMA
raw_data$Brand_sales <- raw_data$HYQ_PI_PATIENTS
raw_data$B_Pat <- raw_data$NUM_PATIENTS_COVERAGE

# Aggregate Promo Channels
raw_data$P_Display   <- raw_data$P_HCP_PLD_IMPRESSIONS + raw_data$P_RXNT_IMPRESSION + raw_data$P_MEDSCAPE_IMPRESSION
raw_data$P_PS        <- raw_data$P_HCP_PS_IMPRESSIONS + raw_data$P_DTC_PS_IMPRESSIONS
raw_data$P_FB        <- raw_data$P_DSE_2 + raw_data$P_FB_2
raw_data$P_HCP_Email <- raw_data$P_RTE + raw_data$P_HQ_EMAILS_HCP 
raw_data$P_Digi_imp  <- raw_data$P_PS + raw_data$P_FB

# ---- 4. Add Trend & Lag BEFORE aggregation ----
raw_data <- raw_data %>% 
  arrange(Trans_Level, Date) %>% 
  group_by(Trans_Level) %>%
  mutate(
    sales_lag_1 = lag(Brand_sales, 1),
    Trend = row_number()
  ) %>% 
  ungroup()



# =========================================================================== #
#  4. APPLY ADSTOCK, POWER & LAG TRANSFORMATIONS
# =========================================================================== #
variables_to_transform <- c("P_TOTAL_PDES", "P_HCP_Email", "P_HQ_EMAILS_DTC", "P_Display", "P_Digi_imp")
adstock_rates <- c(0.25, 0.64, 0.2, 0.15, 0.9)
FR_rates      <- c(0.3, 0.34, 0.5, 0.8, 0.191)
lag_rates     <- c(0, 2, 1, 2, 2)

raw_data <- transformation_function(raw_data, variables_to_transform, adstock_rates, FR_rates, lag_rates)
raw_data[is.na(raw_data)] <- 0


# =========================================================================== #
#  5. AGGREGATION & MODELING DATA PREP
# =========================================================================== #
numeric_cols <- names(raw_data)[sapply(raw_data, is.numeric)]

grouped_data <- raw_data %>%
  dplyr::group_by(Trans_Level, Date) %>%
  dplyr::summarise(across(all_of(numeric_cols), ~sum(.x, na.rm = TRUE)), .groups = "drop")

Model_time <- as.Date("2023-01-01")
Cont_time  <- as.Date("2024-03-01")

modeling_data <- grouped_data %>% filter(Date >= Model_time)

# ---- Feature Columns ----
transformed_vars <- mapply(function(var, ad, fr, lag) {
  paste(var, "ad", ad, "fr", fr, "lag", lag, sep = "_")
}, variables_to_transform, adstock_rates, FR_rates, lag_rates, SIMPLIFY = TRUE)

feature_vars <- c(transformed_vars, "B_Pat")

df_model <- as.data.frame(modeling_data[, c("Brand_sales", feature_vars)])


# =========================================================================== #
#  6. RANDOM FOREST MODEL TRAINING
# =========================================================================== #
set.seed(42)
rf_model <- randomForest(
  Brand_sales ~ ., 
  data = df_model,
  ntree = 1000,
  mtry = 3,
  importance = TRUE
)


# =========================================================================== #
#  7. SHAP VALUE CONTRIBUTIONS
# =========================================================================== #
total_sales_12M <- sum(modeling_data %>% filter(Date >= Cont_time) %>% pull(Brand_sales), na.rm = TRUE)

shap_feature_vars <- setdiff(colnames(df_model), "Brand_sales")

cat("Calculating SHAP contributions...\n")
start_time <- Sys.time()

shap_contributions <- fastshap::explain(
  object = rf_model,
  X = df_model[, shap_feature_vars, drop = FALSE],
  pred_wrapper = function(object, newdata) as.numeric(predict(object, newdata)),
  nsim = 200
)

cat("SHAP computation completed in", round(Sys.time() - start_time, 2), "seconds\n")

# ---- SHAP Summary ----
shap_long <- shap_contributions %>%
  as.data.frame() %>%
  tibble::rownames_to_column("id") %>%
  pivot_longer(cols = -id, names_to = "feature", values_to = "phi")

contrib_summary <- shap_long %>%
  group_by(feature) %>%
  summarise(mean_contribution = mean(phi, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    perc_contribution = mean_contribution / sum(mean_contribution, na.rm = TRUE) * 100,
    volume_12M = round(mean_contribution / sum(mean_contribution, na.rm = TRUE) * total_sales_12M),
    volume_conti_12M = sprintf("%.2f%%", perc_contribution)
  ) %>%
  select(variable = feature, volume_12M, volume_conti_12M)

# ---- Add Intercept ----
intercept_volume <- total_sales_12M - sum(contrib_summary$volume_12M, na.rm = TRUE)
intercept_row <- data.frame(
  variable = "intercept",
  volume_12M = intercept_volume,
  volume_conti_12M = sprintf("%.2f%%", intercept_volume / total_sales_12M * 100)
)

contribution_table <- bind_rows(contrib_summary, intercept_row) %>%
  mutate(total_sales_12M = total_sales_12M) %>%
  arrange(desc(volume_12M))

print(contribution_table)
copy_clipboard(contribution_table)


# =========================================================================== #
#  8. MODEL PERFORMANCE METRICS (R², MAPE)
# =========================================================================== #
df_model$total_Pred <- predict(rf_model, df_model)

rsq_data <- modeling_data %>%
  mutate(total_Pred = df_model$total_Pred) %>%
  group_by(Date) %>%
  summarise(
    Actual = sum(Brand_sales, na.rm = TRUE),
    Predicted = sum(total_Pred, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(MAPE = abs(Actual - Predicted) / Actual)

rsq <- cor(rsq_data$Actual, rsq_data$Predicted)^2 * 100
mape <- mean(rsq_data$MAPE, na.rm = TRUE) * 100

cat("\nMODEL PERFORMANCE:\n")
cat("R-squared:", round(rsq, 2), "%\n")
cat("MAPE:", round(mape, 2), "%\n")

copy_clipboard(rsq_data)


# =========================================================================== #
#  9. Full SHAP Values in Long Format
# =========================================================================== #

baseline <- attr(shap_contributions, "baseline") 
shap_all <- shap_contributions %>% 
  as.data.frame() %>% 
  tibble::rownames_to_column("id") %>% 
  pivot_longer(cols = -id, names_to = "feature", values_to = "phi") %>% 
  mutate(id = as.integer(id)) %>% 
  left_join( 
    df_model %>% tibble::rownames_to_column("id") %>% 
      mutate(id = as.integer(id)), by = "id" ) 

copy_clipboard(shap_all) 

baseline <- attr(shap_contributions, "baseline") 
print(paste("Baseline prediction =", round(baseline, 4))) 
# Sum phi by record (id) 
shap_sum <- shap_all %>% 
  group_by(id) %>% 
  summarise(total_shap = sum(phi, na.rm = TRUE), .groups = "drop") 

# Join with predictions 
verify_decomp <- shap_sum %>% 
  left_join(df_model %>% tibble::rownames_to_column("id") %>% mutate(id = as.integer(id), 
                                                                     total_Pred = predict(rf_model, df_model)), by = "id") %>% 
  mutate( 
    recon_Pred = baseline + total_shap,  # SHAP decomposition 
    diff = total_Pred - recon_Pred       # Difference (should be near 0) 
  ) 
# ---- 3. See results ---- 
# head(verify_decomp, 10) 
copy_clipboard(verify_decomp)


# =========================================================================== #
#  10. PERMUTATION IMPORTANCE (R² DROP METHOD)
# =========================================================================== #
feature_names <- setdiff(colnames(df_model), "Brand_sales")

get_r2 <- function(actual, predicted) cor(actual, predicted, use = "complete.obs")^2

baseline_pred <- predict(rf_model, df_model)
baseline_r2 <- get_r2(df_model$Brand_sales, baseline_pred)
cat("Baseline R²:", round(baseline_r2, 4), "\n")

set.seed(42)
n_repeats <- 100
perm_results <- data.frame()

for (feat in feature_names) {
  r2_shuffled <- replicate(n_repeats, {
    shuffled_data <- df_model
    shuffled_data[[feat]] <- sample(shuffled_data[[feat]])
    pred <- predict(rf_model, shuffled_data)
    get_r2(df_model$Brand_sales, pred)
  })
  
  perm_results <- rbind(perm_results, data.frame(
    Feature = feat,
    Baseline_R2 = baseline_r2,
    Mean_Shuffled_R2 = mean(r2_shuffled),
    Std_Shuffled_R2 = sd(r2_shuffled),
    Mean_Drop = baseline_r2 - mean(r2_shuffled),
    Std_Drop = sd(baseline_r2 - r2_shuffled),
    n_repeats = n_repeats
  ))
}

perm_results <- perm_results %>%
  arrange(desc(Mean_Drop)) %>%
  mutate(
    Importance_Level = case_when(
      Mean_Drop > 0.03 ~ "Very High",
      Mean_Drop > 0.015 ~ "High",
      Mean_Drop > 0.005 ~ "Medium",
      TRUE ~ "Low"
    ),
    Drop_in_R2_Pct = round((Mean_Drop / Baseline_R2) * 100, 2)
  )

print(perm_results)
copy_clipboard(perm_results)


# =========================================================================== #
#  11. RESPONSE CURVE
# =========================================================================== #
generate_response_curve_concave <- function(var_name, df, model, contrib_table,
                                            scaling_factors = seq(0, 2, by = 0.05),
                                            lambda = 0.6) {
  
  contrib_value <- contrib_table %>%
    filter(variable == var_name) %>%
    pull(volume_12M)
  
  if (length(contrib_value) == 0) stop("Variable not found in contribution table")
  
  df_zero <- df
  df_zero[[var_name]] <- 0
  base_pred <- mean(predict(model, df_zero))
  
  results <- data.frame()
  for (s in scaling_factors) {
    df_temp <- df
    df_temp[[var_name]] <- df[[var_name]] * s
    pred <- mean(predict(model, df_temp))
    results <- rbind(results, data.frame(investment_pct = s * 100, predicted_sales_model = pred))
  }
  
  results <- results %>%
    mutate(incremental_sales = predicted_sales_model - base_pred)
  
  model_100 <- results %>% filter(investment_pct == 100) %>% pull(incremental_sales)
  scaling_factor <- contrib_value / model_100
  
  results <- results %>%
    mutate(predicted_sales = incremental_sales * scaling_factor)
  
  hill_raw <- ((results$investment_pct / 100)^lambda) /
    (1 + (results$investment_pct / 100)^lambda)
  
  hill_100 <- hill_raw[which(results$investment_pct == 100)]
  correction_factor <- contrib_value / hill_100
  
  results$hill_scaled <- hill_raw * correction_factor
  return(results)
}


# 1) Resonce curve data for P_TOTAL_PDES_ad_0.25_fr_0.3_lag_0

var_to_test <- "P_TOTAL_PDES_ad_0.25_fr_0.3_lag_0"

response_df <- generate_response_curve_concave(
  var_name = var_to_test,
  df = df_model,
  model = rf_model,
  contrib_table = contribution_table,
  scaling_factors = seq(0, 4, by = 0.05),
  lambda = 0.3
)

copy_clipboard(response_df)


# 2) esonce curve data for P_HCP_Email_ad_0.64_fr_0.34_lag_2

var_to_test <- "P_HCP_Email_ad_0.64_fr_0.34_lag_2"


response_df <- generate_response_curve_concave(
  var_name = var_to_test,
  df = df_model,
  model = rf_model,
  contrib_table = contribution_table,
  scaling_factors = seq(0, 4, by = 0.05),
  lambda = 0.34
)

copy_clipboard(response_df)

# 3) Resonce curve data for P_HQ_EMAILS_DTC_ad_0.2_fr_0.5_lag_1

var_to_test <- "P_HQ_EMAILS_DTC_ad_0.2_fr_0.5_lag_1"

response_df <- generate_response_curve_concave(
  var_name = var_to_test,
  df = df_model,
  model = rf_model,
  contrib_table = contribution_table,
  scaling_factors = seq(0, 4, by = 0.05),
  lambda = 0.5
)

copy_clipboard(response_df)


# 4) Resonce curve data for P_Digi_imp_ad_0.9_fr_0.191_lag_2

var_to_test <- "P_Digi_imp_ad_0.9_fr_0.191_lag_2"

response_df <- generate_response_curve_concave(
  var_name = var_to_test,
  df = df_model,
  model = rf_model,
  contrib_table = contribution_table,
  scaling_factors = seq(0, 4, by = 0.05),
  lambda = 0.191
)

copy_clipboard(response_df)


# 5) Resonce curve data for P_Display_ad_0.15_fr_0.8_lag_2

var_to_test <-"P_Display_ad_0.15_fr_0.8_lag_2"

response_df <- generate_response_curve_concave(
  var_name = var_to_test,
  df = df_model,
  model = rf_model,
  contrib_table = contribution_table,
  scaling_factors = seq(0, 4, by = 0.05),
  lambda = 0.8
)

copy_clipboard(response_df)


# =========================================================================== #
#  12. PLOT RESPONSE CURVE
# =========================================================================== #


ggplot(response_df, aes(x = investment_pct, y = hill_scaled)) +
  geom_line(size = 1.2, color = "blue") +
  geom_point(size = 2) +
  geom_vline(xintercept = 100, linetype = "dashed", color = "red") +
  geom_text(aes(x = 100, y = max(hill_scaled), label = "100% = SHAP volume"), vjust = -0.5) +
  labs(
    title = paste("Concave Response Curve -", var_to_test),
    x = "Investment (%)",
    y = "Incremental Sales (Scaled)"
  ) +
  theme_minimal()


###############################################################################
#                                END OF SCRIPT
###############################################################################

