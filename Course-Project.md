---
title: "Human Activity Recognition Study"
output:
  html_document:
    keep_md: yes
---



## Synopsis

Participants in study were asked to perform different excercises with measurements taken using accelerometers on the belt, forearm, arm, and dumbell. These measurements are then used to predict what excercises are performed.

## Data Processing

The data is read in and split the data into the training and testing sets using a 60% split for training based on the "classe", which is the variable we are trying to predict. Models will be trained on the training set and tested on the testing set in order to prevent overfitting. The holdout set with unknown "classe" which we will later try to predict is also brought in and held until later.


```r
data <- read.table("pmltraining.csv", header = TRUE, sep = ',', quote = '\"', dec='.', na.strings = c('NA', '#DIV/0!', 'NULL', ''))[8:160]
test <- read.table("pmltesting.csv", header = TRUE, sep = ',', quote = '\"', dec = '.', na.strings = c('NA', '#DIV/0!', 'NULL', ''))[8:159]

test[sapply(test, is.logical)] <- lapply(test[sapply(test, is.logical)], as.numeric)

inTrain <- createDataPartition(y=data$classe, p=0.6, list = FALSE)
train <- data[inTrain,]
val <- data[-inTrain,]
```

Next data cleaning steps are performed. Variables with near zero variance are removed and BoxCox transformation and k-nearest neighbors imputation are applied. BoxCox will attempt to normalize the data for PCA and knnImpute will fill in missing values which could be troublesome for some algorithms. The same cleaning and preprocess objects from the training set is applied on the test and holdout set.


```r
keepVar <- row.names(subset(nearZeroVar(train, saveMetrics = TRUE), nzv == FALSE))

cleanTrain <- select(train, one_of(keepVar))
cleanVal <- select(val, one_of(keepVar))
cleanTest <- bind_cols(select(test, one_of(keepVar[1:124])), classe = rep('', times = 20))
```

```
## Warning: Unknown columns: `classe`, `NA`
```

```r
preProc1 <- preProcess(cleanTrain, method = c('BoxCox','knnImpute'))
trainPP1 <- predict(preProc1, cleanTrain)
valPP1 <- predict(preProc1, cleanVal)
testPP1 <- predict(preProc1, cleanTest)
```

PCA is performed. Since there are 150+ variables in the dataset, the selected model will likely overfit the data if the variables are used as is. PCA will create weighted combinations of the variables which will make overfitting less likely. Again the same PCA from the training set is applied on the test and holdout sets in order to get good out of sample accuracy metrics.


```r
preProc2 <- preProcess(trainPP1, method = 'pca', thresh = 0.9)
trainPP2 <- predict(preProc2, trainPP1)
trainPP2 <- bind_cols(trainPP2[,-1], classe = train$classe)
valPP2 <- predict(preProc2, valPP1)
valPP2 <- bind_cols(valPP2[,-1], classe = val$classe)
testPP2 <- predict(preProc2, testPP1)
```

Plot shows the separation of the different "classe" by the first two principle components.


```r
qplot(x=PC1, y=PC2, color=classe, data=trainPP2, geom="point")
```

![](Course-Project_files/figure-html/PCA_plot-1.png)<!-- -->

## Prediction

Random forest and boosting algorithms are applied on the training set. k-fold cross validation is used within the training set with 10 folds so that multiple models can be fitted then tested and the best candidate model chosen.


```r
trainContrl <- trainControl(method = 'cv', number = 10)

modFitRF <- train(classe ~ ., method = 'rf', trControl = trainContrl, data = trainPP2)
modFitBo <- train(classe ~ ., method = 'gbm', trControl = trainContrl, data = trainPP2, verbose = FALSE)
```

The candidate random forest and boosting models are then applied on the testing set to determine the out of sample error. Comparison of the accuracy metrics indicates that the random forest model performs much better than the boosting model.


```r
rfPredVal <- predict(modFitRF, valPP2)
boPredVal <- predict(modFitBo, valPP2)

confusionMatrix(val$classe, rfPredVal)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2105   49   38   29   11
##          B   51 1372   62   18   15
##          C   37   49 1247   21   14
##          D   14   16   99 1145   12
##          E   17   38   27   23 1337
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9184          
##                  95% CI : (0.9122, 0.9244)
##     No Information Rate : 0.2835          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8968          
##                                           
##  Mcnemar's Test P-Value : 2.458e-12       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9465   0.9003   0.8466   0.9264   0.9626
## Specificity            0.9774   0.9769   0.9810   0.9787   0.9837
## Pos Pred Value         0.9431   0.9038   0.9115   0.8904   0.9272
## Neg Pred Value         0.9788   0.9760   0.9651   0.9861   0.9919
## Prevalence             0.2835   0.1942   0.1877   0.1575   0.1770
## Detection Rate         0.2683   0.1749   0.1589   0.1459   0.1704
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9620   0.9386   0.9138   0.9525   0.9732
```

```r
confusionMatrix(val$classe, boPredVal)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1849  104  123  121   35
##          B  125 1078  159   75   81
##          C  132  120 1033   46   37
##          D   65   75  175  928   43
##          E   67  141  107   93 1034
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7548          
##                  95% CI : (0.7451, 0.7643)
##     No Information Rate : 0.2852          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6899          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8262   0.7101   0.6468   0.7348   0.8407
## Specificity            0.9317   0.9305   0.9464   0.9456   0.9383
## Pos Pred Value         0.8284   0.7101   0.7551   0.7216   0.7171
## Neg Pred Value         0.9307   0.9305   0.9129   0.9489   0.9694
## Prevalence             0.2852   0.1935   0.2035   0.1610   0.1568
## Detection Rate         0.2357   0.1374   0.1317   0.1183   0.1318
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.8789   0.8203   0.7966   0.8402   0.8895
```

The below plot shows the spread of the random forest predictions by the first two principle components are whether the predictions are correct.


```r
valPP2$predRight <- rfPredVal==valPP2$classe
qplot(PC1, PC2, colour=predRight, data=valPP2)
```

![](Course-Project_files/figure-html/rfplot-1.png)<!-- -->

The final step attempts to predict the previosuly unknown "classe" on the 20 observation holdout set.


```r
finalModFit <- predict(modFitRF, testPP2)
```

The predictions for the 20 observations are: B, A, A, A, A, E, D, D, A, A, B, C, B, A, E, E, A, B, B, B. The out of sample error is estimated to be:
