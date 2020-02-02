## data preprocession ##
library(tidyverse)

indicator.data <- read.csv("indicators.csv")
data <- read.csv("mf850-finalproject-data.csv")
end.col <- which(str_detect(colnames(data), ".end")) #find the column of end_of_month data
tidydata <- data %>% 
  arrange(compid, Date) %>% #order data by company id & Date
  group_by(compid) %>% #group dataframe by different company id
  mutate_at(.funs = list(adj = ~dplyr::lag(., n = 1, default = NA)),
            .vars = end.col) %>% #shift data within each group (company)
  ungroup() %>% 
  mutate(Industry = as.numeric(factor(Industry))) %>% #transform the Industry feature to numerical data
  select(-end.col, RETMONTH_end, -RETMONTH_end_adj, -compid) %>% 
  drop_na() #drop rows with missing value

# date.lst <- date.sample.size$Date #get date in the table
date.lst <- unique(tidydata$Date) #get date in the table
mon.num <- length(date.lst) #get total length of month

# add label for return direction: 1 for positive return,0 for negative return
tidydata$return_direction = rep(0,dim(tidydata)[1]) 
tidydata$return_direction[tidydata$RETMONTH_end > 0] = 1



## feature selection ##
library(xgboost)

train.num <- 36
test.num <- 10
feature_select_result = matrix(0,9,3)
feature_list = rep(0,9)
i = 1
for (n in c(seq(10,80,10),85)){

# ## IC/IR method
#   feature.df <- tidydata %>%  select(-Date, -RETMONTH_end, -fxusd_start,-realized_vol_spx_end_adj,
#                                      -sentiment_bullish_end_adj,-sentiment_bearish_end_adj,-sentiment_neutral_end_adj,
#                                      -Close_end_adj, -Adj_Close_end_adj,-retmonth_spx_end_adj)
#   corr.df <- matrix(, nrow = mon.num, ncol = dim(feature.df)[2]) %>%
#     data.frame()
#   for(j in 1:mon.num){
#     curr.mon <- date.lst[j]
#     curr.feature <- tidydata %>% filter(Date == curr.mon) %>% select(-Date, -RETMONTH_end, -fxusd_start, -realized_vol_spx_end_adj,
#                                                                      -sentiment_bullish_end_adj,-sentiment_bearish_end_adj,-sentiment_neutral_end_adj,
#                                                                      -Close_end_adj, -Adj_Close_end_adj,-retmonth_spx_end_adj)
#     curr.ret <- tidydata %>% filter(Date == curr.mon) %>% select(RETMONTH_end)
#     corr <- cor(curr.ret, curr.feature)
#     corr.df[j,] <- corr
#   }
#   colnames(corr.df) <- colnames(curr.feature)
#   IC.df <- data.frame(Feature = colnames(feature.df),
#                       absIC = abs(apply(corr.df, 2, mean) / apply(corr.df, 2, sd))) %>%
#     arrange(-absIC)
#   n.variable.select <- IC.df %>% top_n(n) %>% select(Feature)
#   variable.select <- n.variable.select$Feature
#   select_data <- tidydata %>% select(Date, variable.select, RETMONTH_end,return_direction)

## regression, R^2 method
  r2.df <- matrix(, nrow = mon.num, ncol = dim(tidydata)[2]-2) %>%
    data.frame()

  for(j in 1:mon.num){
    curr.mon <- date.lst[j]
    curr.feature1 <- tidydata %>% filter(Date == curr.mon) %>% select(-Date, -RETMONTH_end)
    curr.ret <- tidydata %>% filter(Date == curr.mon) %>% select(RETMONTH_end)
    temp <- t(apply(curr.feature1, 2, function(x.col) summary(lm(curr.ret$RETMONTH_end~x.col))$r.squared))
    r2.df[j,] <- temp
  }

  colnames(r2.df) <- colnames(curr.feature1)
  R2.select.df <- data.frame(Feature = colnames(curr.feature1),
                             R2 = apply(r2.df, 2, mean)) %>%
    arrange(-R2)
  r2.variable.select <- R2.select.df %>% top_n(n) %>% select(Feature)
  r2.variable.select <- r2.variable.select$Feature
  select_data <- tidydata %>% select(Date, r2.variable.select, RETMONTH_end,return_direction)


  # split data:train/test once
  train.mon <- date.lst[1:train.num]
  test.mon <- date.lst[train.num + 1:mon.num]
  train.data <- select_data %>%
    filter(Date %in% train.mon) %>% #filter/select data based on the Date column
    #that satisfies the condition (Date is in train.mon)
    select(-Date,-RETMONTH_end)
  train.y <- train.data %>% select(return_direction)  #select return_direction as training data inputs x
  train.x <- train.data %>% select(-return_direction) #select all data except return_direction as training data response Y
  test.data <-  select_data %>%
    filter(Date %in% test.mon) %>%
    select(-Date,-RETMONTH_end)
  test.y <- test.data %>% select(return_direction)
  test.x <- test.data %>% select(-return_direction)

  # model
  library(xgboost)
  xgb_model <- xgboost(data = as.matrix(train.x),
                       label = as.matrix(train.y),
                       eta = 0.1,
                       max_depth = 6,
                       nrounds = 100,
                       objective = "binary:logistic",
                       verbose=FALSE
  )

  # train/test(ouput prob) accuracy
  pred_train.y = predict(xgb_model, as.matrix(train.data))
  pred_train.y[pred_train.y > 0.5] = 1
  pred_train.y[pred_train.y < 0.5] = 0
  accuracy_train = sum(pred_train.y==train.y)/length(pred_train.y)

  pred_test.y = predict(xgb_model, as.matrix(test.data))
  pred_test.y[pred_test.y > 0.5] = 1
  pred_test.y[pred_test.y < 0.5] = 0
  accuracy_test = sum(pred_test.y==test.y)/length(pred_test.y)

  print(c(n,accuracy_test,accuracy_train))
  feature_select_result[i,] = c(n,accuracy_test,accuracy_train)
  i = i + 1

}

plot(feature_select_result[,1],feature_select_result[,2],type="l",
     xlab = "# of features",ylab = 'accuracy of test data',main = "feature selection by return vs feature regression R^2")




## parameter tuning ##
library(xgboost)

# split data:train/test once
train.num <- 36
test.num <- 10
train.mon <- date.lst[1:train.num]
test.mon <- date.lst[train.num + 1:mon.num]
train.data <- tidydata %>%
  filter(Date %in% train.mon) %>% #filter/select data based on the Date column
  #that satisfies the condition (Date is in train.mon)
  select(-Date,-RETMONTH_end)
train.y <- train.data %>% select(return_direction)  #select return_direction as training data inputs x
train.x <- train.data %>% select(-return_direction) #select all data except return_direction as training data response Y
test.data <-  tidydata %>%
  filter(Date %in% test.mon) %>%
  select(-Date,-RETMONTH_end)
test.y <- test.data %>% select(return_direction)
test.x <- test.data %>% select(-return_direction)


# params_df <- expand.grid(
#   eta = c(0.05,0.1,0.3,0.5),
#   max_depth = c(3,5,7,9),
#   nrounds = c(50,100,300,500)
# )

params_df <- expand.grid(
  eta = c(0.05,0.1,0.2,0.3,0.4,0.5),
  max_depth = c(1,2,3,4,5,6,7,8,9),
  nrounds = c(50,100,200,300,400,500)
)


accuracy_df <- data.frame(matrix(0, nrow=dim(params_df)[1], ncol=2))
colnames(accuracy_df) = c('test_accuracy','train_accuracy')
params_tune_result = cbind(params_df,accuracy_df)

for (i in 1:dim(params_df)[1]){

  # model
  xgb_model <- xgboost(data = as.matrix(train.x),
                       label = as.matrix(train.y),
                       eta = params_df[i,1],
                       max_depth = params_df[i,2],
                       nrounds = params_df[i,3],
                       objective = "binary:logistic",
                       verbose=FALSE
  )

  # train/test(ouput prob) accuracy
  pred_train.y = predict(xgb_model, as.matrix(train.x))
  pred_train.y[pred_train.y > 0.5] = 1
  pred_train.y[pred_train.y < 0.5] = 0
  accuracy_train = sum(pred_train.y==train.y)/length(pred_train.y)

  pred_test.y = predict(xgb_model, as.matrix(test.x))
  pred_test.y[pred_test.y > 0.5] = 1
  pred_test.y[pred_test.y < 0.5] = 0
  accuracy_test = sum(pred_test.y==test.y)/length(pred_test.y)

  params_tune_result[i,4:5] = c(accuracy_test,accuracy_train)
  print(params_tune_result[i,])

}

# write.csv(params_tune_result,'params_tune_result.csv')
# params_tune_result = read.csv(file = "params_tune_result.csv",header=TRUE, sep=",")

library(ggplot2)
tuning.graph <- params_tune_result %>% ggplot(aes(x = max_depth, y = test_accuracy)) +
  geom_line(aes()) +
  facet_grid(eta~nrounds, labeller = label_both) +
  ggtitle(paste("Parameter Tuning of XGBoost")) +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"))

graph1 <- params_tune_result %>% filter(eta == 0.3, max_depth == 3) %>% 
  ggplot(aes(x = nrounds)) +
  geom_line(aes(y = train_accuracy, col = "Train Data")) +
  geom_line(aes(y = test_accuracy*1.25, col = "Test Data")) +
  scale_y_continuous(sec.axis = sec_axis(~./1.25, name = "Test Data Accuracy [%]")) +
  scale_colour_manual("Accuracy", 
                      breaks = c("Test Data", "Train Data"),
                      values = c("#D6604D", "#4393C3")) +
  ggtitle(paste("XGBoost nrounds vs accuracy")) +
  ylab("Train Data Accuracy [%]") +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
        legend.position = c(0.875, 0.15),
        legend.background = element_blank())

graph2 <- params_tune_result %>% filter(nrounds == 100, max_depth == 3) %>% 
  ggplot(aes(x = eta)) +
  geom_line(aes(y = train_accuracy, col = "Train Data")) +
  geom_line(aes(y = test_accuracy*1.2, col = "Test Data")) +
  scale_y_continuous(sec.axis = sec_axis(~./1.2, name = "Test Data Accuracy [%]")) +
  scale_colour_manual("Accuracy", 
                      breaks = c("Test Data", "Train Data"),
                      values = c("#D6604D", "#4393C3")) +
  ggtitle(paste("XGBoost eta vs accuracy")) +
  ylab("Train Data Accuracy [%]") +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
        legend.position = c(0.875, 0.15),
        legend.background = element_blank())

graph3 <- params_tune_result %>% filter(nrounds == 100, eta == 0.3) %>% 
  ggplot(aes(x = max_depth)) +
  geom_line(aes(y = train_accuracy, col = "Train Data")) +
  geom_line(aes(y = test_accuracy*1.4, col = "Test Data")) +
  scale_y_continuous(sec.axis = sec_axis(~./1.4, name = "Test Data Accuracy [%]")) +
  scale_colour_manual("Accuracy", 
                      breaks = c("Test Data", "Train Data"),
                      values = c("#D6604D", "#4393C3")) +
  ggtitle(paste("XGBoost max_depth vs accuracy")) +
  ylab("Train Data Accuracy [%]") +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
        legend.position = c(0.875, 0.15),
        legend.background = element_blank())


tuning.graph
graph1
graph2
graph3




## final model for submition ##

# training data
train.num <- 46
train.mon <- date.lst[1:train.num]
train.data <- tidydata %>%
  filter(Date %in% train.mon) %>% #filter/select data based on the Date column
  #that satisfies the condition (Date is in train.mon)
  select(-Date,-RETMONTH_end)
train.y <- train.data %>% select(return_direction)  #select return_direction as training data inputs x
train.x <- train.data %>% select(-return_direction) #select all data except return_direction as training data response Y

# xgBoost model
library(xgboost)
xgb_model <- xgboost(data = as.matrix(train.x),
                     label = as.matrix(train.y),
                     eta = 0.3,
                     max_depth = 3, 
                     nrounds = 100, 
                     objective = "binary:logistic"
)

xgb.save(xgb_model, 'grow_or_fall_forecast_model')

# feature importance
importance_matrix <- xgb.importance(colnames(train.x), model = xgb_model)
xgb.plot.importance(importance_matrix, rel_to_first = TRUE, top_n = 10, xlab = "Relative importance")




