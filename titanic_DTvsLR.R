library(rpart)
library(rpart.plot)
library(lme4)
library(magrittr)

# simple decision tree on titanic survival ----
data <- read.csv("data/train.csv")
x_test <- read.csv("data/test.csv")

# only look at categories, omit continuous columns (or arbitrary ones e.g. tix)
x_train <- data[, c(3, 5, 7, 8, 12)] 
y_train <- data[,2]

x <- cbind(x_train,y_train)

# grow tree 
fit <- rpart(y_train ~ ., data = x, method="class")

# summarize and plot tree
summary(fit)
print(fit)
prp(fit, extra = 4)

# predict survival in test set
predicted <- predict(fit,x_test)

## Logistic regression w. same data (categories only) -----

fit_lr <- glm(y_train ~ ., data = x, family = binomial(link = "logit"))
summary(fit_lr)

predicted_lr = predict(fit_lr, x_test) # same predictions as decision tree

## Logistic regression w. all data incl. continuous ----------------------------
x_train_all <- data[, c(3, 5:8, 10, 12)] # still omits names and tickets

# clean data (could remove rows with NA, but decided to complete w. dummy data)
# function to replace NAs with column mean
NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))
NA2mean_df <- function(df){
  nums <- unlist(lapply(df, is.numeric)) # index numeric columns
  done <- replace(df, nums, lapply(df[, nums], NA2mean))
  return(done)
}

# function to fill empty character cells with most frequent factor
fill_facs <- function(df) {
  for (i in 1:ncol(df)) {
    levels(df[, i])[levels(df[, i]) == ""] <-
      names(which.max(table(df[, i])))
  }
  return(df)
}


# replace NAs with column mean and replace empty cells with most frequent factor
x_train_all %<>% NA2mean_df() %>% fill_facs()

# fit model
fit_lr_all <- glm(y_train ~ ., data = x_all, family = binomial(link = "logit"))
summary(fit_lr_all)

# predict test set outcomes
# first clean test data
x_test <- NA2mean_df(x_test)
x_test <- fill_facs(x_test)

predicted_lr_all = predict(fit_lr_all, x_test) 


