cat("\014")
rm(list = ls()) # clear global environment
graphics.off() # close all graphics

############ Loading the libraries #######
library(ggplot2)
library(dplyr)
library(caret)
library(imbalance)
library(glmnet)
library(ROSE)
library(pROC)
library(e1071)
library(randomForest)
################general functions################
setwd("C:/Users/yrive/OneDrive/Documentos/Adelphi/Strategic Capstone Project")

noshow <- read.csv("noshow.csv") # Reading the data set
str(noshow) # Checking all variables 
table(is.na(noshow)) # Checking the missing value

df <- subset(noshow, select = -c(PatientId, AppointmentID)) # Deleting No important Data

summary(df) # Checking the variables types 

################Zero and Near Zero Variance predictors################

nzv <- nearZeroVar(df) # Checking the Zero variance in all variables in the data set

print(nzv) # The result shows that variable 11(Alcoholism) and 12 (Handicap) have Zero Variance 

##############################Encoding##################################

# Convert the output variable to a binary variable
df$Show.up <- ifelse(df$Show.up == "yes", 1, 0)

# Label Encoding for categorical variables (Gender)

df$Gender <- factor(df$Gender, levels = c("M", "F"), labels = c(0,1), ordered= TRUE)
df$ScheduledDay <- factor(df$ScheduledDay, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"), labels = c(1,2,3,4,5,6), ordered= TRUE)
df$AppointmentDay <- factor(df$AppointmentDay, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"), labels = c(1,2,3,4,5,6), ordered= TRUE)

########## Replace values 1, 2, and 3 of the Handicap Variable with 1 using ifelse()#########

df$Handicap <- ifelse(df$Handicap %in% c(1, 2, 3), 1, df$Handicap)

#############################################################################

df1 <- df # Saving the data set without ouliers treatment for future evaluation 

############### Plotting all numerical variable to see Outliers #########################

boxplot(df$Age, main= "AGE", ylab = "Value")
outliers_age <- boxplot(df$Age, plot = FALSE)$out
outliers_age  # There are outliers in this variable 

boxplot(df$Calling_time..hour.in.a.day., main= "CALLING TIME", ylab = "Value")
outliers_calling <- boxplot(df$Calling_time..hour.in.a.day., plot = FALSE)$out
outliers_calling # There are not outliers

boxplot(df$Waiting_time..minute., main= "WAITING TIME", ylab = "Value") #*
outliers_waiting <- boxplot(df$Waiting_time..minute., plot = FALSE)$out
outliers_waiting # There are outliers in this variable

boxplot(df$Time_b_appointment..day., main= "TIME BETWEEN APPOINTMENT", ylab = "Value") #*
outliers_time <- boxplot(df$Time_b_appointment..day., plot = FALSE)$out
outliers_time # There are outliers in this variable

boxplot(df$Prior_noshow, main= "PRIOR NO SHOW", ylab = "Value") #*
outliers_prior <- boxplot(df$Prior_noshow, plot = FALSE)$out
outliers_prior # There are outliers in this variable

################Binning Continue Variables to fix the outliers ################   

df$Age <- cut(df$Age, breaks = c(-Inf, 18, 35, 50, Inf), labels = c("0-18", "19-35", "36-50", "51"))
outliers_age <- boxplot(df$Age, plot = FALSE)$out
outliers_age # Checking outliers again 


df$Waiting_time..minute. <- cut(df$Waiting_time..minute., breaks = c(-Inf, 1, 2, 7, 14, 30, Inf),include.lowest = TRUE, labels = c("0-1", "1-2", "2-7", "7-14", "14-30", "31"))
outliers_waiting <- boxplot(df$Waiting_time..minute., plot = FALSE)$out
outliers_waiting # Checking outliers again 


df$Time_b_appointment..day. <- cut(df$Time_b_appointment..day., breaks = c(-Inf, 1, 7, 14, 30, Inf),  include.lowest = TRUE, labels = c("0-1", "1-7", "7-14", "14-30", "31"))
outliers_time <- boxplot(df$Time_b_appointment..day., plot = FALSE)$out
outliers_time # Checking outliers again 

df$Prior_noshow <- cut(df$Prior_noshow, breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1), include.lowest = TRUE, labels = c("0", "1", "2", "3", "4"))
outliers_prior <- boxplot(df$Prior_noshow, plot = FALSE)$out
outliers_prior  # Checking outliers again 


# Plot the Prior No show variable in a bar plot to see why the outliers persist

bin_counts <- table(df$Prior_noshow)
barplot(bin_counts, xlab = "Bins", ylab = "Counts", main = "Prior No Show  Binned ")

# NOTE: We can see from the barplot that the values in the range from 1 to 4 are significantly smaller than 0 which means that all values > 0 
#in the original data set are considered outliers.To avoid deleting these values we decide to replace all values > 0 to 1 as well as we did with the handicap variable.  

df$Prior_noshow <- ifelse(df$Prior_noshow %in% c(1, 2, 3, 4), 1, df$Handicap)
outliers_prior <- boxplot(df$Prior_noshow, plot = FALSE)$out
outliers_prior

bin_counts <- table(df$Prior_noshow)
barplot(bin_counts, xlab = "Bins", ylab = "Counts", main = "Prior No Show Binned ")

# Now, we deleted all outliers 

str(df) # checking all variables again to see the transformation of each  
table(is.na(df)) # checking missing value again 


## Transforming the variables type before balancing 

df$Gender <- as.numeric(factor(df$Gender))
df$Age <- as.numeric(factor(df$Age))
df$ScheduledDay <- as.numeric(factor(df$ScheduledDay))
df$AppointmentDay <- as.numeric(factor(df$AppointmentDay))
df$Month <- as.numeric(factor(df$Month))
df$Waiting_time..minute. <- as.numeric(factor(df$Waiting_time..minute.))
df$Time_b_appointment..day. <- as.numeric(factor(df$Time_b_appointment..day.))

str(df) # The data set is ready fo Balancing 

########### Creating training and testing set ###########################
  
trainIndex <- createDataPartition(df$Show.up, p=0.7, list=F)

train <- df[trainIndex, ]
test<- df[-c(trainIndex),]

# Plotting the Output before balancing

table(train$Show.up)
counts <- table(train$Show.up)
barplot(counts, xlab = "Value", ylab = "Nro Patients", main = "Show Up")



###### SCENARIO 1 


# Balance the train data with SMOTE 


train_balanced <- oversample(train, ratio = 0.75, method = "SMOTE", classAttr = "Show.up")

table(train_balanced$Show.up) # Checking the distribution of the balanced output 

# Plotting the balanced Output 
counts <- table(train_balanced$Show.up)
barplot(counts, xlab = "Value", ylab = "Nro Patients", main = "Show Up")


# Feature Selection with LASSO

# Fit a Lasso model using the glmnet function
lasso.model <- glmnet(as.matrix(train_balanced[, -c(16)]), train_balanced$Show.up, alpha = 1)

# Plot the Lasso coefficient path
plot(lasso.model)

# Use cross-validation to select the optimal lambda value
a<-as.matrix(train_balanced[, -c(16)])
cv.lasso <- cv.glmnet(as.matrix(train_balanced[, -c(16)]), train_balanced$Show.up, alpha = 1)
best.lambda <- cv.lasso$lambda.min

# Fit a Lasso model using the optimal lambda value
lasso.model <- glmnet(as.matrix(train_balanced[, -c(16)]), train_balanced$Show.up, alpha = 1, lambda = best.lambda)

# Get the coefficients for the optimal lambda value
lasso.coef <- coef(lasso.model, s = best.lambda)

# Print the coefficients for each variable
print(lasso.coef)


# Bases on lasso result we selected Age, Gender, AppointmentDay, Month, Calling_time..hour.in.a.day, Waiting_time..minute, Financial_aid, Hypertension
# Diabetes, Alcoholism, Handicap, SMS_received, Time_b_appointment..day, Prior_noshow variebles to run our 3 models 


# ANALYTICAL MODELS  

#1. LOGISTIC REGRESSION MODEL
# Fit a logistic regression model using the training set
model_log <- glm(Show.up ~ Age + Gender + AppointmentDay + Month + Calling_time..hour.in.a.day. +
                           Waiting_time..minute. + Financial_aid + Hypertension + Diabetes + Alcoholism +
                          Handicap +  SMS_received + Time_b_appointment..day. + Prior_noshow,  data = train_balanced, family = "binomial")

# Make predictions on the testing set
predictions_log <- predict(model_log, newdata = test, type = "response")

# Convert the predicted class labels and actual class labels to factors with the same levels
predicted_labels_factor_log <- factor(ifelse(predictions_log > 0.5, 1, 0), levels = levels(factor(test$Show.up)))
actual_labels_factor_log <- factor(test$Show.up, levels = levels(predicted_labels_factor_log))

# Compute the confusion matrix
confusion_matrix_log <- confusionMatrix(predicted_labels_factor_log, actual_labels_factor_log)

# Compute the performance metrics
sensitivity_log <- confusion_matrix_log$byClass["Sensitivity"]
specificity_log <- confusion_matrix_log$byClass["Specificity"]
precision_log <- confusion_matrix_log$byClass["Precision"]
gmean_log <- sqrt(sensitivity_log * specificity_log)
accuracy_log <- confusion_matrix_log$overall["Accuracy"]
roc_log <- roc(actual_labels_factor_log, as.numeric(predictions_log))
auc_log <- auc(roc_log)

plot(roc_log)

# Printing the values of the metrics 

cat("Sensitivity: ", sensitivity_log, "\n")  # Sensitivity:   0.647
cat("Specificity: ", specificity_log, "\n")  # Specificity:   0.798
cat("Precision: ", precision_log, "\n")      # Precision:     0.457  
cat("G-Mean: ", gmean_log, "\n")             # G-Mean:        0.719
cat("Accuracy: ", accuracy_log, "\n")        # Accuracy:      0.767  
cat("AUC: ", auc_log, "\n")                  # AUC:           0.807 



###### SCENARIO 2


# Balance the train data with SMOTE

# Feature Selection with LASSO

# ANALYTICAL MODEL - NAIVE BAYES MODEL


# Train a Naive Bayes model using the training set
model_nai <- naiveBayes(Show.up ~ Age + Gender + AppointmentDay + Month + Calling_time..hour.in.a.day. +
                          Waiting_time..minute. + Financial_aid + Hypertension + Diabetes + Alcoholism +
                          Handicap +  SMS_received + Time_b_appointment..day. + Prior_noshow, data = train_balanced)

# Make predictions on the testing set
predictions_nai<- predict(model_nai, newdata = test)

# Convert the predicted values and actual values to factors with the same levels
predicted_factors_nai <- factor(predictions_nai, levels = c("0", "1"))
actual_factors_nai <- factor(test$Show.up, levels = c("0", "1"))

# Compute the confusion matrix
confusion_matrix_nai <- confusionMatrix(predicted_factors_nai, actual_factors_nai)

# Compute the performance metrics
sensitivity_nai <- confusion_matrix_nai$byClass["Sensitivity"]
specificity_nai <- confusion_matrix_nai$byClass["Specificity"]
precision_nai <- confusion_matrix_nai$byClass["Precision"]
gmean_nai <- sqrt(sensitivity_nai * specificity_nai)
accuracy_nai <- confusion_matrix_nai$overall["Accuracy"]
roc_nai <- roc(as.numeric(actual_factors_nai) - 1, as.numeric(predicted_factors_nai) - 1)
auc_nai <- auc(roc_nai)

plot(roc_nai)

# Print the values of all the performance metrics
cat("Sensitivity: ", sensitivity_nai, "\n")     # Sensitivity:  0.726 
cat("Specificity: ", specificity_nai, "\n")     # Specificity:  0.613 
cat("Precision: ", precision_nai, "\n")         # Precision:    0.329
cat("G-Mean: ", gmean_nai, "\n")                # G-Mean:       0.667
cat("Accuracy: ", accuracy_nai, "\n")           # Accuracy:     0.636
cat("AUC: ", auc_nai, "\n")                     # AUC:          0.669 



###### SCENARIO 3


# Balance the train data with SMOTE

# Feature Selection with LASSO

# ANALYTICAL MODEL - RANDOM FOREST MODEL 


train_random <- train_balanced
test_randon <- test 

train_random$Show.up <- factor(train_random$Show.up)
test_randon$Show.up <- factor(test$Show.up)


# Fit a random forest model using the training set
model_rand <- randomForest(Show.up ~ Age + Gender + AppointmentDay + Month + Calling_time..hour.in.a.day. +
                             Waiting_time..minute. + Financial_aid + Hypertension + Diabetes + Alcoholism +
                             Handicap +  SMS_received + Time_b_appointment..day. + Prior_noshow, data = train_random, importance = TRUE)

# Make predictions on the testing set
predictions_rand <- predict(model_rand, newdata = test_randon)

# Convert the predicted values and actual values to factors with the same levels
predicted_factors_rand <- factor(predictions_rand, levels = c("0", "1"))
actual_factors_rand <- factor(test_randon$Show.up, levels = c("0", "1"))

# Compute the confusion matrix
confusion_matrix_rand <- confusionMatrix(predicted_factors_rand, actual_factors_rand)

# Compute the performance metrics
sensitivity_rand <- confusion_matrix_rand$byClass["Sensitivity"]
specificity_rand <- confusion_matrix_rand$byClass["Specificity"]
precision_rand <- confusion_matrix_rand$byClass["Precision"]
gmean_rand <- sqrt(sensitivity_rand * specificity_rand)
accuracy_rand <- confusion_matrix_rand$overall["Accuracy"]
roc_rand <- roc(as.numeric(actual_factors_rand) - 1, as.numeric(predicted_factors_rand) - 1)
auc_rand <- auc(roc_rand)

plot(roc_rand)

# Print the values of all the performance metrics
cat("Sensitivity: ", sensitivity_rand, "\n")  # Sensitivity:  0.528
cat("Specificity: ", specificity_rand, "\n")  # Specificity:  0.894
cat("Precision: ", precision_rand, "\n")      # Precision:    0.566
cat("G-Mean: ", gmean_rand, "\n")             # G-Mean:       0.687 
cat("Accuracy: ", accuracy_rand, "\n")        # Accuracy:     0.818
cat("AUC: ", auc_rand, "\n")                  # AUC:          0.711



###### SCENARIO 4


# Balance Method = SMOTE 


# FEATURE SELECTION WITH PEARSON's CORRELATION

# Select only the numeric variables
num.vars <-train_balanced[, sapply(train_balanced, is.numeric)]

# Calculate the Pearson correlation coefficients
cor.mat <- cor(num.vars)

# Print the correlation matrix
print(cor.mat)

# GETTING THE BEST 10 VARIABLES 
# Get the absolute values of the correlation coefficients
abs.cor <- abs(cor.mat[,"Show.up"])

# Sort the absolute values in descending order and get the indices
top10.index <- order(abs.cor, decreasing = TRUE)[1:11]


# Get the names of the top 5 variables
top10.vars <- names(abs.cor[top10.index])

# Print the top 5 variables
print(top10.vars)

# From PEARSON's CORRELATION results we can select the following variables: 10 variables
# Age, Gender, Waiting_time..minute., Calling_time..hour.in.a.day., Alcoholism, Diabetes, 
# SMS_received, Prior_noshow, Financial_aid, Hypertension


# ANALYTICAL MODEL - LOGISTIC REGRESSION MODEL

# Fit a logistic regression model using the training set
model_log <- glm(Show.up ~ Age + Gender + Calling_time..hour.in.a.day. +
                   Waiting_time..minute. + Financial_aid + Hypertension + Diabetes + Alcoholism +
                   SMS_received + Prior_noshow,  data = train_balanced, family = "binomial")

# Make predictions on the testing set
predictions_log <- predict(model_log, newdata = test, type = "response")

# Convert the predicted class labels and actual class labels to factors with the same levels
predicted_labels_factor_log <- factor(ifelse(predictions_log > 0.5, 1, 0), levels = levels(factor(test$Show.up)))
actual_labels_factor_log <- factor(test$Show.up, levels = levels(predicted_labels_factor_log))

# Compute the confusion matrix
confusion_matrix_log <- confusionMatrix(predicted_labels_factor_log, actual_labels_factor_log)

# Compute the performance metrics
sensitivity_log <- confusion_matrix_log$byClass["Sensitivity"]
specificity_log <- confusion_matrix_log$byClass["Specificity"]
precision_log <- confusion_matrix_log$byClass["Precision"]
gmean_log <- sqrt(sensitivity_log * specificity_log)
accuracy_log <- confusion_matrix_log$overall["Accuracy"]
roc_log <- roc(actual_labels_factor_log, as.numeric(predictions_log))
auc_log <- auc(roc_log)

plot(roc_log)

# Printing the values of the metrics 

cat("Sensitivity: ", sensitivity_log, "\n")  # Sensitivity:   0.615  
cat("Specificity: ", specificity_log, "\n")  # Specificity:   0.795  
cat("Precision: ", precision_log, "\n")      # Precision:     0.440  
cat("G-Mean: ", gmean_log, "\n")             # G-Mean:        0.670  
cat("Accuracy: ", accuracy_log, "\n")        # Accuracy:      0.758  
cat("AUC: ", auc_log, "\n")                  # AUC:           0.788  



###### SCENARIO 5


# Balance Method = SMOTE 


# FEATURE SELECTION WITH PEARSON's CORRELATION


# ANALYTICAL MODEL - NAIVE BAYES MODEL


# Train a Naive Bayes model using the training set
model_nai <- naiveBayes(Show.up ~ Age + Gender + Calling_time..hour.in.a.day. +
                          Waiting_time..minute. + Financial_aid + Hypertension + Diabetes + Alcoholism +
                          SMS_received + Prior_noshow, data = train_balanced)



# Make predictions on the testing set
predictions_nai<- predict(model_nai, newdata = test)

# Convert the predicted values and actual values to factors with the same levels
predicted_factors_nai <- factor(predictions_nai, levels = c("0", "1"))
actual_factors_nai <- factor(test$Show.up, levels = c("0", "1"))


# Compute the confusion matrix
confusion_matrix_nai <- confusionMatrix(predicted_factors_nai, actual_factors_nai)

# Compute the performance metrics
sensitivity_nai <- confusion_matrix_nai$byClass["Sensitivity"]
specificity_nai <- confusion_matrix_nai$byClass["Specificity"]
precision_nai <- confusion_matrix_nai$byClass["Precision"]
gmean_nai <- sqrt(sensitivity_nai * specificity_nai)
accuracy_nai <- confusion_matrix_nai$overall["Accuracy"]
roc_nai <- roc(as.numeric(actual_factors_nai) - 1, as.numeric(predicted_factors_nai) - 1)
auc_nai <- auc(roc_nai)

plot(roc_nai)

# Print the values of all the performance metrics
cat("Sensitivity: ", sensitivity_nai, "\n")     # Sensitivity:  0.715
cat("Specificity: ", specificity_nai, "\n")     # Specificity:  0.628 
cat("Precision: ", precision_nai, "\n")         # Precision:    0.334
cat("G-Mean: ", gmean_nai, "\n")                # G-Mean:       0.670
cat("Accuracy: ", accuracy_nai, "\n")           # Accuracy:     0.646
cat("AUC: ", auc_nai, "\n")                     # AUC:          0.671 



###### SCENARIO 6

# Balance Method = SMOTE 

# FEATURE SELECTION WITH PEARSON's CORRELATION

# ANALYTICAL MODEL - RANDOM FOREST MODEL 

train_random <- train_balanced
test_randon <- test 

train_random$Show.up <- factor(train_random$Show.up)
test_randon$Show.up <- factor(test$Show.up)


# Fit a random forest model using the training set
model_rand <- randomForest(Show.up ~ Age + Gender + Calling_time..hour.in.a.day. +
                             Waiting_time..minute. + Financial_aid + Hypertension + Diabetes + Alcoholism +
                             SMS_received + Prior_noshow, data = train_random, importance = TRUE)

# Make predictions on the testing set
predictions_rand <- predict(model_rand, newdata = test_randon)

# Convert the predicted values and actual values to factors with the same levels
predicted_factors_rand <- factor(predictions_rand, levels = c("0", "1"))
actual_factors_rand <- factor(test_randon$Show.up, levels = c("0", "1"))

# Compute the confusion matrix
confusion_matrix_rand <- confusionMatrix(predicted_factors_rand, actual_factors_rand)

# Compute the performance metrics
sensitivity_rand <- confusion_matrix_rand$byClass["Sensitivity"]
specificity_rand <- confusion_matrix_rand$byClass["Specificity"]
precision_rand <- confusion_matrix_rand$byClass["Precision"]
gmean_rand <- sqrt(sensitivity_rand * specificity_rand)
accuracy_rand <- confusion_matrix_rand$overall["Accuracy"]
roc_rand <- roc(as.numeric(actual_factors_rand) - 1, as.numeric(predicted_factors_rand) - 1)
auc_rand <- auc(roc_rand)

plot(roc_rand)

# Print the values of all the performance metrics
cat("Sensitivity: ", sensitivity_rand, "\n")  # Sensitivity:  0.507
cat("Specificity: ", specificity_rand, "\n")  # Specificity:  0.900
cat("Precision: ", precision_rand, "\n")      # Precision:    0.572
cat("G-Mean: ", gmean_rand, "\n")             # G-Mean:       0.676 
cat("Accuracy: ", accuracy_rand, "\n")        # Accuracy:     0.819
cat("AUC: ", auc_rand, "\n")                  # AUC:          0.704


###### SCENARIO 7

# Balance the train data with ADASYN 

train_balanced_ADASYN <- oversample(train, ratio = 0.8, method = "ADASYN", classAttr = "Show.up")

table(train_balanced_ADASYN$Show.up)

counts_1 <- table(train_balanced_ADASYN$Show.up)
barplot(counts_1, xlab = "Value", ylab = "Nro Patients", main = "Show Up")


# Feature Selection with PEARSON's CORRELATION

num.vars <-train_balanced_ADASYN[, sapply(train_balanced_ADASYN, is.numeric)]

cor.mat <- cor(num.vars)

print(cor.mat)

abs.cor <- abs(cor.mat[,"Show.up"])

top10.index <- order(abs.cor, decreasing = TRUE)[1:11]

top10.vars <- names(abs.cor[top10.index])

print(top10.vars)


# ANALYTICAL MODEL - RANDOM FOREST MODEL

# LOGISTIC REGRESSION MODEL
model_log <- glm(Show.up ~ Age + Calling_time..hour.in.a.day. +
                   Waiting_time..minute. + Financial_aid + Hypertension + Diabetes + Alcoholism  +
                   SMS_received + Prior_noshow + Month,  data = train_balanced_ADASYN, family = "binomial")


# Make predictions on the testing set
predictions_log <- predict(model_log, newdata = test, type = "response")

# Convert the predicted class labels and actual class labels to factors with the same levels
predicted_labels_factor_log <- factor(ifelse(predictions_log > 0.5, 1, 0), levels = levels(factor(test$Show.up)))
actual_labels_factor_log <- factor(test$Show.up, levels = levels(predicted_labels_factor_log))

# Compute the confusion matrix
confusion_matrix_log <- confusionMatrix(predicted_labels_factor_log, actual_labels_factor_log)

# Compute the performance metrics
sensitivity_log <- confusion_matrix_log$byClass["Sensitivity"]
specificity_log <- confusion_matrix_log$byClass["Specificity"]
precision_log <- confusion_matrix_log$byClass["Precision"]
gmean_log <- sqrt(sensitivity_log * specificity_log)
accuracy_log <- confusion_matrix_log$overall["Accuracy"]
roc_log <- roc(actual_labels_factor_log, as.numeric(predictions_log))
auc_log <- auc(roc_log)

plot(roc_log)

# Printing the values of the metrics /

cat("Sensitivity: ", sensitivity_log, "\n")  # Sensitivity:   0.654 
cat("Specificity: ", specificity_log, "\n")  # Specificity:   0.773
cat ("Precision: ", precision_log, "\n")     # Precision:     0.431
cat("G-Mean: ", gmean_log, "\n")             # G-Mean:        0.711
cat("Accuracy: ", accuracy_log, "\n")        # Accuracy:      0.748
cat("AUC: ", auc_log, "\n")                  # AUC:           0.787



###### SCENARIO 8


# Balance the train data with ADASYN

# Feature Selection with PEARSON's CORRELATION

# ANALYTICAL MODEL - NAIVE BAYES MODEL

# Train a Naive Bayes model using the training set
model_nai <- naiveBayes(Show.up ~ Age + Calling_time..hour.in.a.day. +
                          Waiting_time..minute. + Financial_aid + Hypertension + Diabetes + Alcoholism  +
                          SMS_received + Prior_noshow + Month, data = train_balanced_ADASYN)


# Make predictions on the testing set
predictions_nai<- predict(model_nai, newdata = test)

# Convert the predicted values and actual values to factors with the same levels
predicted_factors_nai <- factor(predictions_nai, levels = c("0", "1"))
actual_factors_nai <- factor(test$Show.up, levels = c("0", "1"))


# Compute the confusion matrix
confusion_matrix_nai <- confusionMatrix(predicted_factors_nai, actual_factors_nai)

# Compute the performance metrics
sensitivity_nai <- confusion_matrix_nai$byClass["Sensitivity"]
specificity_nai <- confusion_matrix_nai$byClass["Specificity"]
precision_nai <- confusion_matrix_nai$byClass["Precision"]
gmean_nai <- sqrt(sensitivity_nai * specificity_nai)
accuracy_nai <- confusion_matrix_nai$overall["Accuracy"]
roc_nai <- roc(as.numeric(actual_factors_nai) - 1, as.numeric(predicted_factors_nai) - 1)
auc_nai <- auc(roc_nai)

plot(roc_nai)

# Print the values of all the performance metrics
cat("Sensitivity: ", sensitivity_nai, "\n")     # Sensitivity:  0.738
cat("Specificity: ", specificity_nai, "\n")     # Specificity:  0.592
cat("Precision: ", precision_nai, "\n")         # Precision:    0.322
cat("G-Mean: ", gmean_nai, "\n")                # G-Mean:       0.661
cat("Accuracy: ", accuracy_nai, "\n")           # Accuracy:     0.622
cat("AUC: ", auc_nai, "\n")                     # AUC:          0.665


###### SCENARIO 9


# Balance the train data with ADASYN

# Feature Selection with PEARSON's CORRELATION

# ANALYTICAL MODEL - RANDOM FOREST MODEL 

train_random_ADASYN <- train_balanced_ADASYN
test_randon <- test 

train_random_ADASYN$Show.up <- factor(train_random_ADASYN$Show.up)
test_randon$Show.up <- factor(test$Show.up)

# Fit a random forest model using the training set
model_rand <- randomForest(Show.up ~ Age + Calling_time..hour.in.a.day. +
                             Waiting_time..minute. + Financial_aid + Hypertension + Diabetes + Alcoholism  +
                             SMS_received + Prior_noshow + Month, data = train_random_ADASYN, importance = TRUE)

# Make predictions on the testing set
predictions_rand <- predict(model_rand, newdata = test_randon)

# Convert the predicted values and actual values to factors with the same levels
predicted_factors_rand <- factor(predictions_rand, levels = c("0", "1"))
actual_factors_rand <- factor(test_randon$Show.up, levels = c("0", "1"))

# Compute the confusion matrix
confusion_matrix_rand <- confusionMatrix(predicted_factors_rand, actual_factors_rand)

# Compute the performance metrics
sensitivity_rand <- confusion_matrix_rand$byClass["Sensitivity"]
specificity_rand <- confusion_matrix_rand$byClass["Specificity"]
precision_rand <- confusion_matrix_rand$byClass["Precision"]
gmean_rand <- sqrt(sensitivity_rand * specificity_rand)
accuracy_rand <- confusion_matrix_rand$overall["Accuracy"]
roc_rand <- roc(as.numeric(actual_factors_rand) - 1, as.numeric(predicted_factors_rand) - 1)
auc_rand <- auc(roc_rand)

plot(roc_rand)

# Print the values of all the performance metrics
cat("Sensitivity: ", sensitivity_rand, "\n")  # Sensitivity:  0.531
cat("Specificity: ", specificity_rand, "\n")  # Specificity:  0.883
cat("Precision: ", precision_rand, "\n")      # Precision:    0.544
cat("G-Mean: ", gmean_rand, "\n")             # G-Mean:       0.685
cat("Accuracy: ", accuracy_rand, "\n")        # Accuracy:     0.810
cat("AUC: ", auc_rand, "\n")                  # AUC:          0.707



# Based on all metric we selected as our best scenario, the scenario 1 which has the best G-Mean score.
# Having selected  the scenario 1 as the best scenario we will use it to evaluate the most important feature for the model
# Also, we will use the same scenario to analyze the impact of the outliers in the performance of the model 


############ LOOKING FOR THE MOST IMPORTANT FEATURE TO THE MODEL ##############3#


# Create a data frame with the feature names and their coefficients from feature selection in the scenario 1

feature_coefficients <- data.frame(
  Feature = c("Age", "Gender", "ScheduledDay", "AppointmentDay", "Month",
              "Calling_time..hour.in.a.day.", "Waiting_time..minute.",
              "Financial_aid", "Hypertension", "Diabetes", "Alcoholism",
              "Handicap", "SMS_received", "Time_b_appointment..day.", "Prior_noshow"),
  Coefficient = c(0.0113675859, -0.0153679177, 0.0010284474, 0.0006579669, 0.0557396460,
                  0.0003226186, -0.0794999147, 0.0665903564, 0.0462396674, 0.0379682296,
                  0.0725628274, 0.3381568039, 0.0269356814, 0.0628899991, -0.4879071846)
)


# Create a bar plot using ggplot2
ggplot(feature_coefficients, aes(x = reorder(Feature, Coefficient), y = Coefficient, fill = Coefficient)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Lasso Model Feature Coefficients",
       x = "Features",
       y = "Coefficient") +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0) +
  coord_flip()

# From the plot we can see that the Handicap and Prior_noshow variables are the most important feature for our model.
# Handicap has a positive correlation with our output Show.up which mean that being handicapped is associated with
# a considerable increase of Show.up.
# Prior_noshow has a negative correlation with Show.up which indicates that having prior no-shows is associated with a 
# considerable decrease in the Show.up.



####### CHECKING THE IMPACT OF OUTLIERS TO THE MODEL ##############

str(df1) # checking the original data set with outliers

### Transforming factor data into numerical data before balancing

df1$Gender <- as.numeric(factor(df1$Gender))

df1$ScheduledDay <- as.numeric(factor(df1$ScheduledDay))

df1$AppointmentDay <- as.numeric(factor(df1$AppointmentDay))


########### Creating training and testing set ###########################

trainIndex_out <- createDataPartition(df1$Show.up, p=0.7, list=F)

train_out <- df1[trainIndex_out, ]
test_out<- df1[-c(trainIndex_out),]


# Balance the train data with SMOTE 

train_balanced_out <- oversample(train_out, ratio = 0.75, method = "SMOTE", classAttr = "Show.up")

table(train_balanced_out$Show.up) # Checking the distribution of the balanced output 

# Plotting the balanced Output 
counts_out <- table(train_balanced_out$Show.up)
barplot(counts_out, xlab = "Value", ylab = "Nro Patients", main = "Show Up")

# Feature Selection with LASSO

# Fit a Lasso model using the glmnet function
lasso.model_out <- glmnet(as.matrix(train_balanced_out[, -c(16)]), train_balanced_out$Show.up, alpha = 1)

# Plot the Lasso coefficient path
plot(lasso.model_out)

# Use cross-validation to select the optimal lambda value
a_out<-as.matrix(train_balanced_out[, -c(16)])
cv.lasso_out <- cv.glmnet(as.matrix(train_balanced_out[, -c(16)]), train_balanced_out$Show.up, alpha = 1)
best.lambda_out <- cv.lasso_out$lambda.min

# Fit a Lasso model using the optimal lambda value
lasso.model_out <- glmnet(as.matrix(train_balanced_out[, -c(16)]), train_balanced_out$Show.up, alpha = 1, lambda = best.lambda_out)

# Get the coefficients for the optimal lambda value
lasso.coef_out <- coef(lasso.model_out, s = best.lambda_out)

# Print the coefficients for each variable
print(lasso.coef_out)


# ANALYTICAL MODELS  

#1. LOGISTIC REGRESSION MODEL
# Fit a logistic regression model using the training set
model_log_out <- glm(Show.up ~ Age + Gender + AppointmentDay + Month + Calling_time..hour.in.a.day. +
                   Waiting_time..minute. + Financial_aid + Hypertension + Diabetes + Alcoholism +
                   SMS_received + Time_b_appointment..day. + Prior_noshow,  data = train_balanced_out, family = "binomial")

# Make predictions on the testing set
predictions_log_out <- predict(model_log_out, newdata = test_out, type = "response")

# Convert the predicted class labels and actual class labels to factors with the same levels
predicted_labels_factor_log_out <- factor(ifelse(predictions_log_out > 0.5, 1, 0), levels = levels(factor(test_out$Show.up)))
actual_labels_factor_log_out <- factor(test_out$Show.up, levels = levels(predicted_labels_factor_log_out))

# Compute the confusion matrix
confusion_matrix_log_out <- confusionMatrix(predicted_labels_factor_log_out, actual_labels_factor_log_out)

# Compute the performance metrics
sensitivity_log_out <- confusion_matrix_log_out$byClass["Sensitivity"]
specificity_log_out <- confusion_matrix_log_out$byClass["Specificity"]
precision_log_out <- confusion_matrix_log_out$byClass["Precision"]
gmean_log_out <- sqrt(sensitivity_log_out * specificity_log_out)
accuracy_log_out <- confusion_matrix_log_out$overall["Accuracy"]
roc_log_out <- roc(actual_labels_factor_log_out, as.numeric(predictions_log_out))
auc_log_out <- auc(roc_log_out)

plot(roc_log_out)

# Printing the values of the metrics                              Outliers     No Outliers 

cat("Sensitivity: ", sensitivity_log_out, "\n")  # Sensitivity:     0.543       0.647
cat("Specificity: ", specificity_log_out, "\n")  # Specificity:     0.854       0.798
cat("Precision: ", precision_log_out, "\n")      # Precision:       0.501       0.457
cat("G-Mean: ", gmean_log_out, "\n")             # G-Mean:          0.681       0.719
cat("Accuracy: ", accuracy_log_out, "\n")        # Accuracy:        0.788       0.767 
cat("AUC: ", auc_log_out, "\n")                  # AUC:             0.787       0.807

# We can see from the results that the presence of outliers have a significant impact in the performance of the model. 
# The metrics show how well the model works without outliers


