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

set.seed(1946)

pa.org <- read.csv("People Analytics at Teach For America Data Set.csv")
str(pa.org)
summary(pa.org)
head(pa.org)
sum(is.na(pa.org))

#removing unneccesary columns
pa.df <- pa.org [, c(-1, -2, -5, -6, -7, -8, -(18:27), -(35:37), -39) ]
head(pa.df)
str(pa.df)
summary(pa.df)


## behavioral subset 
pa_behavioral <- pa.df[, c(-(1:7), -17) ]

#deriving columns
pa_behavioral$Total.Essay.Length <- pa_behavioral$Essay.1.Length + pa_behavioral$Essay.2.Length + pa_behavioral$Essay.3.Length

#converting to date type
pa_behavioral$Sign.up.Date         <- as.Date(pa_behavioral$Sign.up.Date,         format = "%d-%m-%Y")
pa_behavioral$Started.Date         <- as.Date(pa_behavioral$Started.Date,         format = "%d-%m-%Y")
pa_behavioral$Submitted.Date       <- as.Date(pa_behavioral$Submitted.Date,       format = "%d-%m-%Y")
pa_behavioral$Application.Deadline <- as.Date(pa_behavioral$Application.Deadline, format = "%d-%m-%Y")

#deriving columns
pa_behavioral$Days.to.Start   <- as.numeric(pa_behavioral$Started.Date   - pa_behavioral$Sign.up.Date)
pa_behavioral$Days.to.Submit  <- as.numeric(pa_behavioral$Submitted.Date - pa_behavioral$Started.Date)
pa_behavioral$Deadline.Gap    <- as.numeric(pa_behavioral$Application.Deadline - pa_behavioral$Submitted.Date)

#dropping columns not required 
cols_to_remove <- match(c("Sign.up.Date", "Started.Date", 
                          "Application.Deadline", "Submitted.Date"),
                        names(pa_behavioral))
pa_behavioral <- pa_behavioral[, -cols_to_remove]

#removing data with negative values
neg_cols <- c("Days.to.Start","Days.to.Submit","Deadline.Gap")
for (v in neg_cols) pa_behavioral[[v]][pa_behavioral[[v]] < 0] <- NA
pa_behavioral <- tidyr::drop_na(pa_behavioral, all_of(neg_cols))

#converting outcome variable to factor 
pa_behavioral$Completed.Admissions.Process <- as.factor(pa_behavioral$Completed.Admissions.Process)

str(pa_behavioral)


##behavior partition 

behavior.idx <- createDataPartition(pa_behavioral$Completed.Admissions.Process, p=0.8, list = FALSE )

train.behavior.df <-pa_behavioral[behavior.idx,]
test.behavior.df <-pa_behavioral[-behavior.idx,]

#handling class imbalance
train.behavior.df <- ovun.sample(
  Completed.Admissions.Process ~ .,
  data   = train.behavior.df,
  method = "both",   # hybrid oversampling + undersampling
  p      = 0.5,      # target 50/50 balance
  seed   = 1946
)$data

train.behavior.df$Completed.Admissions.Process <- as.factor(train.behavior.df$Completed.Admissions.Process)
test.behavior.df$Completed.Admissions.Process  <- as.factor(test.behavior.df$Completed.Admissions.Process)


## Check class proportions in each subset
prop.table(table(pa_behavioral$Completed.Admissions.Process))
prop.table(table(train.behavior.df$Completed.Admissions.Process))


#MODEL 1: K-NEAREST NEIGHBORS (KNN) 
modelLookup("knn")

model_knn <- train(Completed.Admissions.Process ~ .,
                        data      = train.behavior.df,
                        method    = "knn",
                        tuneLength = 10,
                        preProcess = c("center", "scale"),
                        trControl = trainControl(method = "cv", number = 10))

model_knn$bestTune
confusionMatrix(predict(model_knn, test.behavior.df), 
                test.behavior.df$Completed.Admissions.Process, 
                positive = "1")


#MODEL 2: NAIVE BAYES (NB) 
modelLookup('naive_bayes')

tune_grid_nb <- expand.grid(
  laplace   = c(0,0.5, 1, 1.5, 2),
  usekernel = c(TRUE, FALSE),
  adjust    = c(0.75, 1, 1.25, 1.5,1.75,2)
)

model_nb <- train(Completed.Admissions.Process ~ .,
                        data      = train.behavior.df,
                        method    = "naive_bayes",
                        tuneGrid  = tune_grid_nb,
                        preProcess = c("center", "scale"),
                        trControl = trainControl(method = "cv", number = 10))

model_nb$bestTune
confusionMatrix(predict(model_nb, test.behavior.df), 
                test.behavior.df$Completed.Admissions.Process, 
                positive = "1")


#MODEL 3: DECISION TRESS (DT)
modelLookup('C5.0')

tune_grid_dt <- expand.grid(
  model  = c("rules", "tree"),
  winnow = TRUE,
  trials = seq(1, 9, 1)
)

model_dt <- train(Completed.Admissions.Process ~ ., 
                  data = train.behavior.df,
                  method = "C5.0",
                  tuneGrid = tune_grid_dt,
                  preProcess = c("center", "scale"), 
                  trControl = trainControl(method = "cv", number = 10))

model_dt$bestTune
confusionMatrix(predict(model_dt, test.behavior.df), 
                test.behavior.df$Completed.Admissions.Process, 
                positive = "1")


#MODEL 4: ARTIFICIAL NEURAL NETWORKS (ANN)
modelLookup("nnet")

tune_grid_ann <- expand.grid(size = seq(5,15,5), 
                             decay = seq(0.1,0.7, 0.2))
                       
model_ann <- train(Completed.Admissions.Process ~ . ,
                   data=train.behavior.df,
                   method = "nnet",
                   tuneGrid = tune_grid_ann,
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "cv", number = 5), 
                   trace = FALSE,
                   metric = "Kappa", 
                   maxit = 50)

 model_ann$bestTune
confusionMatrix(predict(model_ann, test.behavior.df), 
                test.behavior.df$Completed.Admissions.Process, 
                positive = "1")

plotnet(model_ann$finalModel)


#MODEL 5: SUPPORT VECTOR MACHINE (SVM)
modelLookup('svmRadial')

tune_grid_svm <- expand.grid(
  sigma = c(0.01),
  C     = c(1, 2)
)

model_svm <- train(Completed.Admissions.Process ~ . ,
                      data = train.behavior.df,
                      method = "svmRadial",
                      tuneGrid = tune_grid_svm,
                      preProcess = c("center", "scale"),
                      trControl = trainControl(method = "cv", number = 3))

model_svm$bestTune
confusionMatrix(predict(model_svm, test.behavior.df), 
                test.behavior.df$Completed.Admissions.Process, 
                positive = "1")







