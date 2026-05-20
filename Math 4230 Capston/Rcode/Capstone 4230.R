library(tidyverse)
library(tidymodels)
library(car)
library(MASS)
library(glmnet)
library(randomForest)
library(gbm)
library(e1071)
library(nnet)
library(caret)

setwd("C:/Users/rener/Downloads")

housing <- read.csv("housing.csv")



glimpse(housing)
summary(housing)
colSums(is.na(housing))

housing <- housing %>%
  mutate(
    high_value = ifelse(
      median_house_value > median(median_house_value, na.rm = TRUE),
      1, 0),high_value = as.factor(high_value))

housing <- housing %>%
  mutate(
    rooms_per_household = total_rooms / households,
    bedrooms_per_room = total_bedrooms / total_rooms,
    population_per_household = population / households
  )

# get rid of any NA 

housing <- housing %>%
  mutate(
    total_bedrooms = ifelse(
      is.na(total_bedrooms),
      median(total_bedrooms, na.rm = TRUE),
      total_bedrooms
    )
  )


# Train & Test Split 


set.seed(4230)

train_index <- createDataPartition(
  housing$high_value,
  p = 0.8,
  list = FALSE
)

train_data <- housing[train_index, ]
test_data  <- housing[-train_index, ]



#Chapter 2 


library(tidyverse)
library(caret)
library(GGally)
library(corrplot)
library(gridExtra)



housing <- read.csv("housing.csv")

glimpse(housing)
summary(housing)


colSums(is.na(housing))


housing$total_bedrooms[is.na(housing$total_bedrooms)] <- 
  median(housing$total_bedrooms, na.rm = TRUE)


median_price <- median(housing$median_house_value)

housing$high_value <- ifelse(
  housing$median_house_value > median_price,
  1,
  0
)

housing$high_value <- as.factor(housing$high_value)



set.seed(4230)

train_index <- createDataPartition(
  housing$high_value,
  p = 0.8,
  list = FALSE
)

train_data <- housing[train_index, ]
test_data  <- housing[-train_index, ]



summary_stats <- housing %>%
  select_if(is.numeric) %>%
  summarise_all(list(
    mean = ~round(mean(., na.rm = TRUE), 0),
    sd = ~sd(., na.rm = TRUE),
    median = ~median(., na.rm = TRUE),
    min = ~min(., na.rm = TRUE),
    max = ~max(., na.rm = TRUE)
  ))

summary_stats
# Histograms

ggplot(housing, aes(median_house_value)) +
  geom_histogram(bins = 40) +
  theme_minimal()

# Box Plots  

ggplot(housing, aes(ocean_proximity, median_house_value)) +
  geom_boxplot() +
  theme_minimal()


# Scatter Plots 


ggplot(housing,
       aes(median_income, median_house_value)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm") +
  theme_minimal()

# Correlation Matrix


numeric_data <- housing %>%
  select_if(is.numeric)

cor_matrix <- cor(numeric_data)

corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.cex = 0.7)



write.csv(train_data,
          "train_data.csv",
          row.names = FALSE)

write.csv(test_data,
          "test_data.csv",
          row.names = FALSE)








housing <- housing %>%
  mutate(
    ocean_proximity = as.factor(ocean_proximity),
    rooms_per_household = total_rooms / households,
    bedrooms_per_room = total_bedrooms / total_rooms,
    population_per_household = population / households
  )

set.seed(4230)

train_index <- createDataPartition(
  housing$high_value,
  p = 0.8,
  list = FALSE
)

train_data <- housing[train_index, ]
test_data  <- housing[-train_index, ]


# HELPER FUNCTIONS

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

mae <- function(actual, predicted) {
  mean(abs(actual - predicted))
}

regression_metrics <- function(actual, predicted) {
  data.frame(
    RMSE = rmse(actual, predicted),
    MAE = mae(actual, predicted),
    R2 = cor(actual, predicted)^2
  )
}


# CHAPTER 3: SIMPLE LINEAR REGRESSION

slr_model <- lm(
  median_house_value ~ median_income,
  data = train_data
)

summary(slr_model)

slr_pred <- predict(slr_model, newdata = test_data)

slr_results <- regression_metrics(
  actual = test_data$median_house_value,
  predicted = slr_pred
)

slr_results

ggplot(train_data, aes(x = median_income, y = median_house_value)) +
  geom_point(alpha = 0.25) +
  geom_smooth(method = "lm", se = FALSE) +
  theme_minimal() +
  labs(
    title = "Simple Linear Regression: Median Income vs House Value",
    x = "Median Income",
    y = "Median House Value"
  )

par(mfrow = c(2, 2))
plot(slr_model)
par(mfrow = c(1, 1))


# CHAPTER 4: MULTIPLE LINEAR REGRESSION

mlr_model <- lm(
  median_house_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data
)

summary(mlr_model)

mlr_pred <- predict(mlr_model, newdata = test_data)

mlr_results <- regression_metrics(
  actual = test_data$median_house_value,
  predicted = mlr_pred
)

mlr_results

# Dummy variables
dummy_matrix <- model.matrix(mlr_model)
head(dummy_matrix)

# VIF
vif_values <- vif(mlr_model)
vif_values

# Stepwise subset selection using AIC
step_model <- stepAIC(
  mlr_model,
  direction = "both",
  trace = FALSE
)

summary(step_model)

step_pred <- predict(step_model, newdata = test_data)

step_results <- regression_metrics(
  actual = test_data$median_house_value,
  predicted = step_pred
)

step_results

# AIC and BIC
AIC(mlr_model)
BIC(mlr_model)

AIC(step_model)
BIC(step_model)

model_comparison_ch4 <- rbind(
  SLR = slr_results,
  MLR = mlr_results,
  Stepwise_MLR = step_results
)

model_comparison_ch4


# CHAPTER 5: LOGISTIC REGRESSION

logistic_model <- glm(
  high_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data,
  family = binomial
)

summary(logistic_model)

logistic_prob <- predict(
  logistic_model,
  newdata = test_data,
  type = "response"
)

logistic_pred <- ifelse(logistic_prob > 0.5, 1, 0)
logistic_pred <- as.factor(logistic_pred)

confusionMatrix(
  logistic_pred,
  test_data$high_value,
  positive = "1"
)


# CHAPTER 6: CROSS-VALIDATION

train_control <- trainControl(
  method = "cv",
  number = 10
)

cv_lm_model <- train(
  median_house_value ~ median_income,
  data = train_data,
  method = "lm",
  trControl = train_control
)

cv_lm_model

cv_mlr_model <- train(
  median_house_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data,
  method = "lm",
  trControl = train_control
)

cv_mlr_model


# CHAPTER 6: BOOTSTRAP CONFIDENCE INTERVAL

bootstrap_income_coef <- function(data, indices) {
  sample_data <- data[indices, ]
  model <- lm(median_house_value ~ median_income, data = sample_data)
  return(coef(model)[2])
}

set.seed(4230)

boot_results <- boot(
  data = train_data,
  statistic = bootstrap_income_coef,
  R = 1000
)

boot_results

boot.ci(
  boot_results,
  type = "perc"
)

# CHAPTER 7: RIDGE AND LASSO REGRESSION

x_train <- model.matrix(
  median_house_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data
)[, -1]

y_train <- train_data$median_house_value

x_test <- model.matrix(
  median_house_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = test_data
)[, -1]

y_test <- test_data$median_house_value


# RIDGE REGRESSION

set.seed(4230)

ridge_cv <- cv.glmnet(
  x_train,
  y_train,
  alpha = 0,
  nfolds = 10
)

plot(ridge_cv)

best_lambda_ridge <- ridge_cv$lambda.min
best_lambda_ridge

ridge_model <- glmnet(
  x_train,
  y_train,
  alpha = 0,
  lambda = best_lambda_ridge
)

ridge_pred <- predict(
  ridge_model,
  s = best_lambda_ridge,
  newx = x_test
)

ridge_results <- regression_metrics(
  actual = y_test,
  predicted = as.numeric(ridge_pred)
)

ridge_results


# LASSO REGRESSION

set.seed(4230)

lasso_cv <- cv.glmnet(
  x_train,
  y_train,
  alpha = 1,
  nfolds = 10
)

plot(lasso_cv)

best_lambda_lasso <- lasso_cv$lambda.min
best_lambda_lasso

lasso_model <- glmnet(
  x_train,
  y_train,
  alpha = 1,
  lambda = best_lambda_lasso
)

lasso_pred <- predict(
  lasso_model,
  s = best_lambda_lasso,
  newx = x_test
)

lasso_results <- regression_metrics(
  actual = y_test,
  predicted = as.numeric(lasso_pred)
)

lasso_results

# Lasso selected variables
lasso_coefficients <- coef(lasso_model)
lasso_coefficients


# COMPARE REGRESSION MODELS SO FAR

model_comparison_regression <- rbind(
  SLR = slr_results,
  MLR = mlr_results,
  Stepwise_MLR = step_results,
  Ridge = ridge_results,
  Lasso = lasso_results
)

model_comparison_regression




# CHAPTER 8: DECISION TREES

library(rpart)
library(rpart.plot)

# 8A. REGRESSION TREE

reg_tree <- rpart(
  median_house_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data,
  method = "anova"
)

printcp(reg_tree)
plotcp(reg_tree)

rpart.plot(reg_tree)

reg_tree_pred <- predict(reg_tree, newdata = test_data)

reg_tree_results <- regression_metrics(
  actual = test_data$median_house_value,
  predicted = reg_tree_pred
)

reg_tree_results


# 8B. CLASSIFICATION TREE

class_tree <- rpart(
  high_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data,
  method = "class"
)

printcp(class_tree)
plotcp(class_tree)

rpart.plot(class_tree)

class_tree_pred <- predict(
  class_tree,
  newdata = test_data,
  type = "class"
)

confusionMatrix(
  class_tree_pred,
  test_data$high_value,
  positive = "1"
)


# CHAPTER 9: TREE ENSEMBLES

# 9A. BAGGING

set.seed(4230)

bagging_model <- randomForest(
  median_house_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data,
  mtry = 12,
  ntree = 100,
  importance = TRUE
)

bagging_model

bagging_pred <- predict(
  bagging_model,
  newdata = test_data
)

bagging_results <- regression_metrics(
  actual = test_data$median_house_value,
  predicted = bagging_pred
)

bagging_results

# OOB error
plot(bagging_model)

# Variable importance
importance(bagging_model)
varImpPlot(bagging_model)


# 9B. RANDOM FOREST

set.seed(4230)

rf_model <- randomForest(
  median_house_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data,
  ntree = 300,
  importance = TRUE
)

rf_model

rf_pred <- predict(
  rf_model,
  newdata = test_data
)

rf_results <- regression_metrics(
  actual = test_data$median_house_value,
  predicted = rf_pred
)

rf_results

plot(rf_model)

importance(rf_model)
varImpPlot(rf_model)


# 9C. BOOSTING

set.seed(4230)

boost_model <- gbm(
  median_house_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data,
  distribution = "gaussian",
  n.trees = 3000,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  verbose = FALSE
)

best_trees <- gbm.perf(
  boost_model,
  method = "cv"
)

best_trees

boost_pred <- predict(
  boost_model,
  newdata = test_data,
  n.trees = best_trees
)

boost_results <- regression_metrics(
  actual = test_data$median_house_value,
  predicted = boost_pred
)

boost_results

summary(boost_model)


# CHAPTER 10: SUPPORT VECTOR MACHINES

# 10A. LINEAR SVM

set.seed(4230)

svm_linear_model <- svm(
  high_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data,
  kernel = "linear",
  scale = TRUE
)

svm_linear_pred <- predict(
  svm_linear_model,
  newdata = test_data
)

confusionMatrix(
  svm_linear_pred,
  test_data$high_value,
  positive = "1"
)


# 10B. RBF SVM

set.seed(4230)

svm_rbf_model <- svm(
  high_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data,
  kernel = "radial",
  scale = TRUE
)

svm_rbf_pred <- predict(
  svm_rbf_model,
  newdata = test_data
)

confusionMatrix(
  svm_rbf_pred,
  test_data$high_value,
  positive = "1"
)


# CHAPTER 11: K-NEAREST NEIGHBORS

library(class)

# Create dummy variables for categorical predictors
knn_train_x <- model.matrix(
  high_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data
)[, -1]

knn_test_x <- model.matrix(
  high_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = test_data
)[, -1]

# Standardize predictors
preprocess_knn <- preProcess(
  knn_train_x,
  method = c("center", "scale")
)

knn_train_scaled <- predict(
  preprocess_knn,
  knn_train_x
)

knn_test_scaled <- predict(
  preprocess_knn,
  knn_test_x
)

# Try different K values
k_values <- c(3, 5, 7, 9, 11, 15, 21)

knn_results <- data.frame(
  K = numeric(),
  Accuracy = numeric()
)

for (k in k_values) {
  
  knn_pred <- knn(
    train = knn_train_scaled,
    test = knn_test_scaled,
    cl = train_data$high_value,
    k = k
  )
  
  cm <- confusionMatrix(
    knn_pred,
    test_data$high_value,
    positive = "1"
  )
  
  knn_results <- rbind(
    knn_results,
    data.frame(
      K = k,
      Accuracy = cm$overall["Accuracy"]
    )
  )
}

knn_results

ggplot(knn_results, aes(x = K, y = Accuracy)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(
    title = "KNN Accuracy Across Different K Values",
    x = "Number of Neighbors (K)",
    y = "Test Accuracy"
  )




# CHAPTER 8: DECISION TREES

library(rpart)
library(rpart.plot)

# 8A. REGRESSION TREE

reg_tree <- rpart(
  median_house_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data,
  method = "anova"
)

printcp(reg_tree)
plotcp(reg_tree)

rpart.plot(reg_tree)

reg_tree_pred <- predict(reg_tree, newdata = test_data)

reg_tree_results <- regression_metrics(
  actual = test_data$median_house_value,
  predicted = reg_tree_pred
)

reg_tree_results


# 8B. CLASSIFICATION TREE

class_tree <- rpart(
  high_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data,
  method = "class"
)

printcp(class_tree)
plotcp(class_tree)

rpart.plot(class_tree)

class_tree_pred <- predict(
  class_tree,
  newdata = test_data,
  type = "class"
)

confusionMatrix(
  class_tree_pred,
  test_data$high_value,
  positive = "1"
)



# CHAPTER 12: PRINCIPAL COMPONENT ANALYSIS

pca_data <- housing %>%
  dplyr::select_if(is.numeric) %>%
  dplyr::select(-median_house_value)
pca_scaled <- scale(pca_data)

# Run PCA
pca_model <- prcomp(
  pca_scaled,
  center = TRUE,
  scale. = TRUE
)

summary(pca_model)

# Scree plot
plot(
  pca_model,
  type = "l",
  main = "Scree Plot for Principal Component Analysis"
)

# Proportion of variance explained
pca_variance <- pca_model$sdev^2 / sum(pca_model$sdev^2)

pca_variance_table <- data.frame(
  Principal_Component = paste0("PC", 1:length(pca_variance)),
  Variance_Explained = pca_variance,
  Cumulative_Variance = cumsum(pca_variance)
)

pca_variance_table

# Loadings
pca_loadings <- pca_model$rotation
pca_loadings

# First two principal components
pca_scores <- as.data.frame(pca_model$x)

pca_scores$high_value <- housing$high_value

ggplot(pca_scores, aes(x = PC1, y = PC2, color = high_value)) +
  geom_point(alpha = 0.4) +
  theme_minimal() +
  labs(
    title = "PCA Plot: First Two Principal Components",
    x = "Principal Component 1",
    y = "Principal Component 2"
  )


# CHAPTER 13: NEURAL NETWORK


nn_train_x <- model.matrix(
  high_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = train_data
)[, -1]

nn_test_x <- model.matrix(
  high_value ~ longitude + latitude + housing_median_age +
    total_rooms + total_bedrooms + population + households +
    median_income + ocean_proximity +
    rooms_per_household + bedrooms_per_room +
    population_per_household,
  data = test_data
)[, -1]

# Scale predictors
preprocess_nn <- preProcess(
  nn_train_x,
  method = c("center", "scale")
)

nn_train_scaled <- predict(
  preprocess_nn,
  nn_train_x
)

nn_test_scaled <- predict(
  preprocess_nn,
  nn_test_x
)

# Convert to data frames
nn_train_data <- as.data.frame(nn_train_scaled)
nn_test_data <- as.data.frame(nn_test_scaled)

# Add response variable
nn_train_data$high_value <- train_data$high_value
nn_test_data$high_value <- test_data$high_value

# Fit neural network classification model
set.seed(4230)

nn_model <- nnet(
  high_value ~ .,
  data = nn_train_data,
  size = 5,
  decay = 0.01,
  maxit = 300,
  trace = FALSE
)

nn_model

# Predict probabilities
nn_prob <- predict(
  nn_model,
  newdata = nn_test_data,
  type = "raw"
)

# Convert probabilities to class predictions
nn_pred <- ifelse(nn_prob > 0.5, 1, 0)
nn_pred <- as.factor(nn_pred)

# Confusion matrix
nn_cm <- confusionMatrix(
  nn_pred,
  nn_test_data$high_value,
  positive = "1"
)

nn_cm


# CHAPTER 14: METHOD COMPARISON AND FINAL RECOMMENDATIONS

# 14A. REGRESSION METHOD COMPARISON


model_comparison_regression_final <- rbind(
  SLR = slr_results,
  MLR = mlr_results,
  Stepwise_MLR = step_results,
  Ridge = ridge_results,
  Lasso = lasso_results,
  Regression_Tree = reg_tree_results,
  Bagging = bagging_results,
  Random_Forest = rf_results,
  Boosting = boost_results
)

model_comparison_regression_final

# Sort by lowest RMSE
model_comparison_regression_final_sorted <- model_comparison_regression_final %>%
  arrange(RMSE)

model_comparison_regression_final_sorted


# 14B. CLASSIFICATION METHOD COMPARISON

# Extract accuracy values from confusion matrices.

logistic_cm <- confusionMatrix(
  logistic_pred,
  test_data$high_value,
  positive = "1"
)

class_tree_cm <- confusionMatrix(
  class_tree_pred,
  test_data$high_value,
  positive = "1"
)

svm_linear_cm <- confusionMatrix(
  svm_linear_pred,
  test_data$high_value,
  positive = "1"
)

svm_rbf_cm <- confusionMatrix(
  svm_rbf_pred,
  test_data$high_value,
  positive = "1"
)

best_k <- knn_results$K[which.max(knn_results$Accuracy)]

best_knn_pred <- knn(
  train = knn_train_scaled,
  test = knn_test_scaled,
  cl = train_data$high_value,
  k = best_k
)

best_knn_cm <- confusionMatrix(
  best_knn_pred,
  test_data$high_value,
  positive = "1"
)

classification_comparison <- data.frame(
  Method = c(
    "Logistic Regression",
    "Classification Tree",
    "Linear SVM",
    "RBF SVM",
    "KNN",
    "Neural Network"
  ),
  Accuracy = c(
    logistic_cm$overall["Accuracy"],
    class_tree_cm$overall["Accuracy"],
    svm_linear_cm$overall["Accuracy"],
    svm_rbf_cm$overall["Accuracy"],
    best_knn_cm$overall["Accuracy"],
    nn_cm$overall["Accuracy"]
  ),
  Sensitivity = c(
    logistic_cm$byClass["Sensitivity"],
    class_tree_cm$byClass["Sensitivity"],
    svm_linear_cm$byClass["Sensitivity"],
    svm_rbf_cm$byClass["Sensitivity"],
    best_knn_cm$byClass["Sensitivity"],
    nn_cm$byClass["Sensitivity"]
  ),
  Specificity = c(
    logistic_cm$byClass["Specificity"],
    class_tree_cm$byClass["Specificity"],
    svm_linear_cm$byClass["Specificity"],
    svm_rbf_cm$byClass["Specificity"],
    best_knn_cm$byClass["Specificity"],
    nn_cm$byClass["Specificity"]
  )
)

classification_comparison

classification_comparison_sorted <- classification_comparison %>%
  arrange(desc(Accuracy))

classification_comparison_sorted


# 14C. INTERPRETABILITY RATING

method_summary_table <- data.frame(
  Method = c(
    "Simple Linear Regression",
    "Multiple Linear Regression",
    "Stepwise Multiple Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Regression Tree",
    "Bagging",
    "Random Forest",
    "Boosting",
    "Logistic Regression",
    "Classification Tree",
    "Linear SVM",
    "RBF SVM",
    "KNN",
    "PCA",
    "Neural Network"
  ),
  Task = c(
    "Regression",
    "Regression",
    "Regression",
    "Regression",
    "Regression",
    "Regression",
    "Regression",
    "Regression",
    "Regression",
    "Classification",
    "Classification",
    "Classification",
    "Classification",
    "Classification",
    "Unsupervised",
    "Classification"
  ),
  Interpretability = c(
    "High",
    "High",
    "High",
    "Medium",
    "Medium",
    "Medium",
    "Low",
    "Medium",
    "Low",
    "High",
    "Medium",
    "Medium",
    "Low",
    "Medium",
    "Medium",
    "Low"
  ),
  Notes = c(
    "Easy to explain but limited to one predictor.",
    "More complete than SLR and still interpretable.",
    "Improves simplicity by removing less useful predictors.",
    "Controls coefficient instability but less direct to interpret.",
    "Performs variable selection while controlling overfitting.",
    "Captures nonlinear splits and is visually interpretable.",
    "Improves tree accuracy but reduces interpretability.",
    "Strong accuracy with useful variable importance.",
    "Often strong predictive performance but less transparent.",
    "Interpretable classification probabilities and odds.",
    "Visual classification rules are easy to explain.",
    "Clearer than RBF SVM but still less direct than logistic regression.",
    "Flexible nonlinear classifier but difficult to interpret.",
    "Simple idea but sensitive to scaling and K.",
    "Helpful for dimensionality reduction and structure discovery.",
    "Flexible model but least interpretable."
  )
)

method_summary_table