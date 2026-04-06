# Load libraries
library(class)
library(e1071)
library(nnet)
library(rpart)
library(caret)
library(dplyr)
library(fastDummies)
library(NeuralNetTools)

set.seed(1946)

pa.org <- read.csv("People_Analytics.csv")
str(pa.org)
summary(pa.org)
head(pa.org)
sum(is.na(pa.org))

#removing unnecessary columns
pa.df <- pa.org [, c(-1, -2, -5, -6, -7, -8, -(18:27), -(35:37), -39) ]
head(pa.df)
str(pa.df)
summary(pa.df)

#academic subset
pa_academic <- pa.df[, c(-(8:16), -18)]

#deriving columns
pa_academic$major_count <- apply(
  pa_academic[, c("Major.1...Cleaned", "Major.2...Cleaned", "Minor...Cleaned")],
  1,
  function(x) sum(x != "none" & x != "" & !is.na(x))
)

#dropping columns not required 
pa_academic$stem <- pa_academic$Is.Math..Sci..or.Eng.Major.Minor

cols_to_remove <- match(c("Undergraduate.University...Cleaned", 
                          "Major.1...Cleaned", "Major.2...Cleaned", 
                          "Minor...Cleaned", 
                          "Is.Math..Sci..or.Eng.Major.Minor"),
                        names(pa_academic))
cols_to_remove <- cols_to_remove[!is.na(cols_to_remove)]
pa_academic <- pa_academic[, -cols_to_remove]

#converting variables to dummy column & factors 
pa_academic$Completed.Admissions.Process <- as.factor(pa_academic$Completed.Admissions.Process)

pa_academic <- dummy_cols(pa_academic,  select_columns = c("stem", "School.Selectivity", "Region.Preference.Level"), 
                          remove_first_dummy = FALSE, 
                          remove_selected_columns = TRUE)

str(pa_academic)


#academic partition 

academic.idx <- createDataPartition(pa_academic$Completed.Admissions.Process, p=0.8, list = FALSE )

train.academic.df <-pa_academic[academic.idx,]
test.academic.df <-pa_academic[-academic.idx,]

#handling class imbalance
train.academic.df <- ovun.sample(
  Completed.Admissions.Process ~ .,
  data   = train.academic.df,
  method = "both",   # hybrid oversampling + undersampling
  p      = 0.5,      # target 50/50 balance
  seed   = 1946
)$data

train.academic.df$Completed.Admissions.Process <- as.factor(train.academic.df$Completed.Admissions.Process)
test.academic.df$Completed.Admissions.Process  <- as.factor(test.academic.df$Completed.Admissions.Process)


# Check class proportions in each subset
prop.table(table(pa_academic$Completed.Admissions.Process))
prop.table(table(train.academic.df$Completed.Admissions.Process))


#MODEL 1: K-NEAREST NEIGHBORS (KNN) 
model_knn <- train (Completed.Admissions.Process ~ .,
                    data       = train.academic.df,
                    method     = "knn",
                    tuneLength = 10,                  # tests k = 1 to 10
                    preProcess = c("center", "scale"),
                    trControl  = trainControl(method = "cv", number = 10)
                   )

model_knn$bestTune
confusionMatrix(predict(model_knn, test.academic.df), 
                test.academic.df$Completed.Admissions.Process, 
                positive = "1")


#MODEL 2: NAIVE BAYES (NB) 
modelLookup('naive_bayes')

tune_grid_nb <- expand.grid(
  laplace   = c(0,0.5, 1, 1.5, 2),
  usekernel = c(TRUE, FALSE),
  adjust    = c(0.75, 1, 1.25, 1.5,1.75,2)
  )

model_nb <- train(Completed.Admissions.Process ~ .,
                        data       = train.academic.df,
                        method     = "naive_bayes",
                        tuneGrid   = tune_grid_nb,
                        preProcess = c("center", "scale"),
                        trControl  = trainControl(method = "cv", number = 10))

model_nb$bestTune
confusionMatrix(predict(model_nb, test.academic.df), 
                test.academic.df$Completed.Admissions.Process, 
                positive = "1")


#MODEL 3: DECISION TRESS (DT)
modelLookup('C5.0')

tune_grid_dt <- expand.grid(
  model  = c("rules", "tree"),
  winnow = TRUE,
  trials = seq(1, 9, 1)
)

model_dt <- train(Completed.Admissions.Process ~ ., 
                  data = train.academic.df,
                  method = "C5.0",
                  trControl = trainControl(method = "cv", number = 10),
                  preProcess = c("center", "scale"),
                  tuneGrid = tune_grid_dt)

model_dt$bestTune
confusionMatrix(predict(model_dt, test.academic.df), 
                test.academic.df$Completed.Admissions.Process, 
                positive = "1")


#MODEL 4: ARTIFICIAL NEURAL NETWORKS (ANN)
modelLookup("nnet")

tune_grid_ann <- expand.grid(
              size = seq(5,15,5), 
              decay = seq(0.1,0.7, 0.2))

model_ann <- train(Completed.Admissions.Process ~ ., 
                   data = train.academic.df,
                   method = "nnet",
                   metric = "Kappa",         
                   maxit = 50,
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "cv", number = 5),
                   tuneGrid = tune_grid_ann,
                   trace = FALSE)

model_ann$bestTune
confusionMatrix(predict(model_ann, test.academic.df), 
                test.academic.df$Completed.Admissions.Process, 
                positive = "1")

plotnet(model_ann$finalModel)


#MODEL 5: SUPPORT VECTOR MACHINE (SVM)
modelLookup('svmRadial')

tune_grid_svm <- expand.grid(
  sigma = c(0.01),
  C     = c(1, 2)
)

model_svm <- train(Completed.Admissions.Process ~ . ,
                   data = train.academic.df,
                   method = "svmRadial",
                   tuneGrid = tune_grid_svm,
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "cv", number = 3))

model_svm$bestTune
confusionMatrix(predict(model_svm, test.academic.df), 
                test.academic.df$Completed.Admissions.Process, 
                positive = "1")

























