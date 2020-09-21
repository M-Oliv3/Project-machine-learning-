# **Project: Prediction of the manner in which someone do a exercise.**

----
### Author: Maicon

## Description

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

The goal of this project will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, in order to predict the manner in which they did the exercise.

## **Data Processing**

The data for this project come from this source [Data](http://groupware.les.inf.puc-rio.br/har), and the [pml-training](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and [pml-testing](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

#### **Reference**

Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 

### **Reading in the accelerometers data**


### **Load the data**

**We first read the training and testing data**

```{r dataset,echo = TRUE, cache=TRUE}

train<- read.csv("pml-training.csv")
test<- read.csv("pml-testing.csv")

```

**After reading the data we check the first few rows (there are `r nrow(train)`) rows in the Training set and (`r nrow(test)`) in the Testing set.** 


```{r}
summary(train)
```


## **Preprossing**

#### **Select features**

```{r echo = TRUE, cache=TRUE}

library(caret)
train<-train[, 8:160]
test<-test[, 8:160]

```

#### **Delete Zero- and Near Zero-Variance Predictors**

```{r echo = TRUE, cache=TRUE}

nzv <- nearZeroVar(train,freqCut = 2 ,uniqueCut = 20)
filtered.Tr <- train[, -nzv]
filtered.Te <- test[,-nzv]

```

#### **Number of NAs per column**

```{r echo = TRUE, cache=TRUE}

sort(colSums(is.na(train)))

```

#### **Delete predictors with high numbers of Nas**

```{r echo = TRUE, cache=TRUE}

keep.cols.tr = which(apply(!is.na(filtered.Tr), 2, all))
clean.tr = filtered.Tr[,keep.cols.tr]
clean.te = filtered.Te[,keep.cols.tr]

test1 <-clean.te

```


## **Create a building data set and validation set**

```{r echo = TRUE, cache=TRUE}

inBuild <- createDataPartition(y=clean.tr$classe,p=0.7, list=FALSE)
validation<-clean.tr[-inBuild,]
buildData<-clean.tr[inBuild,]
inTrain = createDataPartition(y=buildData$classe, p = 0.7, list=FALSE)

training = buildData[ inTrain,]
testing = buildData[-inTrain,]

```


## **Results**

#### **Random forest model**

```{r echo = TRUE, cache=TRUE}
set.seed(123)

model.rf<-train(classe~.,method="rf",
              data=training,
              trControl=trainControl(method="cv"))
model.rf

```

#### **Prediction random forest**

```{r echo = TRUE, cache=TRUE}

prediction.rf <-predict(model.rf, newdata=testing)

Confusion.rf <- confusionMatrix(testing$classe,prediction.rf)
Accuracy.rf <- Confusion.rf$overall[1]

```

**Random forest = `r Accuracy.rf`**

#### **Boosting model** 

```{r echo = TRUE, cache=TRUE}
library(gbm)

model.gbm <- train(classe~., data=training, method="gbm", verbose=FALSE)

model.gbm
```


#### **Prediction gbm**

```{r echo = TRUE, cache=TRUE}

prediction.gbm <-predict(model.gbm, newdata=testing)

Confusion.gbm <- confusionMatrix(testing$classe,prediction.gbm)
Accuracy.gbm <- Confusion.gbm$overall[1]
Accuracy.gbm
```

**Boosting = `r Accuracy.gbm`**

#### **Fit a model that combines predictors**

```{r echo = TRUE, cache=TRUE}

predDF <- data.frame(prediction.rf, prediction.gbm, classe = testing$classe)
combModFit <- train(classe ~ ., method = "rf", data = predDF)
combPred <- predict(combModFit, predDF)


Confusion.comb <- confusionMatrix(testing$classe,combPred)
Accuracy.comb <- Confusion.comb$overall[1]
```

**Accuracy = `r Accuracy.comb`**

#### **Validation**
```{r echo = TRUE, cache=TRUE}

predV.rf <-predict(model.rf,validation)
predV.gbm <-predict(model.gbm,validation)
predV.comb <-data.frame(prediction.rf=predV.rf, prediction.gbm=predV.gbm)
combPredV <-predict(combModFit,predV.comb)

Confusion.rf.V <- confusionMatrix(validation$classe,predV.rf)
Confusion.gbm.V <- confusionMatrix(validation$classe,predV.gbm)
Confusion.comb.V <- confusionMatrix(validation$classe,combPredV)

Accuracy.rf.V <- Confusion.rf.V$overall[1]
Accuracy.gbm.V <- Confusion.gbm.V$overall[1]
Accuracy.comb.V <- Confusion.comb.V$overall[1]

```

**The out of sample - Accuracy**

Random forest - `r Accuracy.rf.V`

Boosting - `r Accuracy.gbm.V`

Combination - `r Accuracy.comb.V`

#### **Prediction with the testing data from [pml-testing](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)**

```{r echo = TRUE, cache=TRUE}

Test.pred.rf<-predict(model.rf,test1)
Test.pred.gbm<-predict(model.gbm,test1)
Test.predV.comb<-data.frame(prediction.rf=Test.pred.rf,prediction.gbm=Test.pred.gbm)
combPredV <-predict(combModFit,Test.predV.comb)


results<-data.frame(rf=Test.pred.rf,gbm=Test.pred.gbm,comb=combPredV)
results
```

The results showed that the method of combining predictors in this problem did not produce a relevant increase in the accuracy, when it is compared to Random forest.