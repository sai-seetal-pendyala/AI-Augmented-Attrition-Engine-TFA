# Load libraries
library(class)
library(e1071)
library(nnet)
library(rpart)
library(caret)
library(dplyr)
library(ROSE)
library(pROC)
library(fastDummies)
library(NeuralNetTools)
library(tidyr)

set.seed(1946)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
pa.org <- read.csv("People Analytics at Teach For America Data Set.csv")
str(pa.org)
summary(pa.org)
head(pa.org)
sum(is.na(pa.org))

# Remove unnecessary columns
pa.df <- pa.org[, c(-1, -2, -5, -6, -7, -8, -(18:27), -(35:37), -39)]
head(pa.df)
str(pa.df)
summary(pa.df)


# ============================================================================
# STEP 2: CREATE COMBINED DATASET
# Behavioral + Academic variables
# ============================================================================

combined.df <- pa.df  # start from cleaned base dataset

# ----------------------
# Behavioral Derivations
# ----------------------
combined.df$Total.Essay.Length <- combined.df$Essay.1.Length +
                                  combined.df$Essay.2.Length +
                                  combined.df$Essay.3.Length

# Convert dates
combined.df$Sign.up.Date         <- as.Date(combined.df$Sign.up.Date,         format = "%d-%m-%Y")
combined.df$Started.Date         <- as.Date(combined.df$Started.Date,         format = "%d-%m-%Y")
combined.df$Submitted.Date       <- as.Date(combined.df$Submitted.Date,       format = "%d-%m-%Y")
combined.df$Application.Deadline <- as.Date(combined.df$Application.Deadline, format = "%d-%m-%Y")

# Behavioral engineered variables
combined.df$Days.to.Start   <- as.numeric(combined.df$Started.Date   - combined.df$Sign.up.Date)
combined.df$Days.to.Submit  <- as.numeric(combined.df$Submitted.Date - combined.df$Started.Date)
combined.df$Deadline.Gap    <- as.numeric(combined.df$Application.Deadline - combined.df$Submitted.Date)

# Remove raw date columns
combined.df <- combined.df[, -match(c("Sign.up.Date","Started.Date","Submitted.Date","Application.Deadline"),
                                    names(combined.df))]


# ----------------------
# Academic Derivations
# ----------------------
combined.df$major_count <- apply(
  combined.df[, c("Major.1...Cleaned", "Major.2...Cleaned", "Minor...Cleaned")],
  1,
  function(x) sum(x != "none" & x != "" & !is.na(x))
)

# STEM binary variable
combined.df$stem <- combined.df$Is.Math..Sci..or.Eng.Major.Minor

# Remove categorical academic fields not needed after deriving major_count
cols_remove_academic <- match(
  c("Undergraduate.University...Cleaned",
    "Major.1...Cleaned","Major.2...Cleaned",
    "Minor...Cleaned",
    "Is.Math..Sci..or.Eng.Major.Minor"),
  names(combined.df)
)

cols_remove_academic <- cols_remove_academic[!is.na(cols_remove_academic)]
combined.df <- combined.df[, -cols_remove_academic]


# ----------------------
# Remove negative time values
# ----------------------
neg_cols <- c("Days.to.Start", "Days.to.Submit", "Deadline.Gap")
for (v in neg_cols) combined.df[[v]][combined.df[[v]] < 0] <- NA
combined.df <- tidyr::drop_na(combined.df, all_of(neg_cols))


# ----------------------
# Convert outcome variable to factor
# ----------------------
combined.df$Completed.Admissions.Process <- as.factor(combined.df$Completed.Admissions.Process)


# ----------------------
# Dummify academic categorical predictors
# ----------------------
combined.df <- dummy_cols(combined.df,
                          select_columns = c("stem", "School.Selectivity", "Region.Preference.Level"),
                          remove_first_dummy = FALSE,
                          remove_selected_columns = TRUE)

str(combined.df)


# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
comb.idx <- createDataPartition(combined.df$Completed.Admissions.Process, p = 0.8, list = FALSE)

train.combined.df <- combined.df[comb.idx, ]
test.combined.df  <- combined.df[-comb.idx, ]


# ============================================================================
# STEP 4: HANDLE CLASS IMBALANCE (ROSE)
# ============================================================================
train.combined.df <- ovun.sample(
  Completed.Admissions.Process ~ .,
  data = train.combined.df,
  method = "both",
  p = 0.5,
  seed = 1946
)$data

train.combined.df$Completed.Admissions.Process <- as.factor(train.combined.df$Completed.Admissions.Process)
test.combined.df$Completed.Admissions.Process  <- as.factor(test.combined.df$Completed.Admissions.Process)

# Check class proportions
prop.table(table(train.combined.df$Completed.Admissions.Process))
prop.table(table(test.combined.df$Completed.Admissions.Process))


# ============================================================================
# MODEL 1: KNN
# ============================================================================
model_knn <- train(Completed.Admissions.Process ~ .,
                   data = train.combined.df,
                   method = "knn",
                   tuneLength = 10,
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "cv", number = 10))

model_knn$bestTune
confusionMatrix(predict(model_knn, test.combined.df),
                test.combined.df$Completed.Admissions.Process,
                positive = "1")


# ============================================================================
# MODEL 2: NAIVE BAYES
# ============================================================================
tune_grid_nb <- expand.grid(
  laplace   = c(0,0.5,1,1.5,2),
  usekernel = c(TRUE, FALSE),
  adjust    = c(0.75,1,1.25,1.5,1.75,2)
)

model_nb <- train(Completed.Admissions.Process ~ .,
                  data = train.combined.df,
                  method = "naive_bayes",
                  tuneGrid = tune_grid_nb,
                  preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv", number = 10))

model_nb$bestTune
confusionMatrix(predict(model_nb, test.combined.df),
                test.combined.df$Completed.Admissions.Process,
                positive = "1")


# ============================================================================
# MODEL 3: DECISION TREE
# ============================================================================
tune_grid_dt <- expand.grid(
  model  = c("rules","tree"),
  winnow = TRUE,
  trials = seq(1, 9, 1)
)

model_dt <- train(Completed.Admissions.Process ~ .,
                  data = train.combined.df,
                  method = "C5.0",
                  tuneGrid = tune_grid_dt,
                  preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv", number = 10))

model_dt$bestTune
confusionMatrix(predict(model_dt, test.combined.df),
                test.combined.df$Completed.Admissions.Process,
                positive = "1")


# ============================================================================
# MODEL 4: ARTIFICIAL NEURAL NETWORK
# ============================================================================
tune_grid_ann <- expand.grid(
  size  = seq(5,15,5),
  decay = seq(0.1,0.7,0.2)
)

model_ann <- train(Completed.Admissions.Process ~ .,
                   data = train.combined.df,
                   method = "nnet",
                   tuneGrid = tune_grid_ann,
                   metric = "Kappa",
                   maxit = 50,
                   preProcess = c("center","scale"),
                   trControl = trainControl(method = "cv", number = 5),
                   trace = FALSE)

model_ann$bestTune
confusionMatrix(predict(model_ann, test.combined.df),
                test.combined.df$Completed.Admissions.Process,
                positive = "1")

plotnet(model_ann$finalModel)


# ============================================================================
# MODEL 5: SVM
# ============================================================================
tune_grid_svm <- expand.grid(
  sigma = c(0.01),
  C     = c(1, 2)
)

model_svm <- train(Completed.Admissions.Process ~ .,
                   data = train.combined.df,
                   method = "svmRadial",
                   tuneGrid = tune_grid_svm,
                   preProcess = c("center","scale"),
                   trControl = trainControl(method = "cv", number = 3))

model_svm$bestTune
confusionMatrix(predict(model_svm, test.combined.df),
                test.combined.df$Completed.Admissions.Process,
                positive = "1")
