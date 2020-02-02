setwd('~/')
library(xgboost)
library(tidyverse)
library(forecast)
library(lubridate)
library(multiway)

train.num <- 36
test.num <- 10
indicator.data <- read.csv("indicators.csv")
data <- read.csv("mf850-finalproject-data.csv")
end.col <- which(str_detect(colnames(data), ".end")) #find the column of end_of_month data
tidydata <- data %>% 
  arrange(compid, Date) %>% #order data by company id & Date
  group_by(compid) %>% #group dataframe by different company id
  mutate_at(.funs = list(adj = ~dplyr::lag(., n = 1, default = NA)),
            .vars = end.col) %>% #shift data within each group (company)
  ungroup() %>% 
  mutate(Industry = as.numeric(factor(Industry)),
         mon = month(as.Date(Date))) %>% #transform the Industry feature to numerical data
  select(-end.col, RETMONTH_end, -RETMONTH_end_adj, -compid) %>% 
  drop_na() #drop rows with missing value

date.sample.size <- data %>% 
               group_by(Date) %>% #break down dataframe into multiple groups/tables according to date
               tally() #count number of dates in each group (date), (date,num_of_instances) pairs is assigned to date.sample.size
date.lst <- date.sample.size$Date #get date in the table
mon.num <- length(date.lst) #get total length of month
R2.mat <- matrix(, nrow = mon.num-train.num, ncol = 3)
R2.df <- data.frame(R2.mat) %>% `colnames<-`(c("Neural.Net","mod1","mod2")) 

i <- 1
train.mon <- date.lst[i:(i+train.num-1)] #train.mon takes length of 5 with rolling window (ex:1-5,2-6,...,42-46)
test.mon <- date.lst[i+train.num] #test.mon is the next month after the last training month
train.data <- tidydata %>% 
  filter(Date %in% train.mon) %>% #filter / select data based on the Date column 
  #that satisfies the condition (Date is in train.mon)
  select(-Date)
train.y <- train.data %>% select(RETMONTH_end)  #select RETMONTH_end as training data inputs x
train.x <- train.data %>% select(-RETMONTH_end) #select all data except RETMONTH_end as training data response Y
test.data <-  tidydata %>% 
  filter(Date == test.mon) %>% 
  select(-Date)
test.y <- test.data %>% select(RETMONTH_end)
test.x <- test.data %>% select(-RETMONTH_end)

#Cross validation
date.lst <- unique(tidydata$Date)
n <- length(tidydata)

xgb_grid <- expand.grid(
  eta = c(0.01,0.02,0.04,0.06),
  nrounds = c(200,500,750,1000,1500),
  max_depth = c(2, 4, 8, 12, 16)
)

ts_cv_mse <- function(params,nrounds,stp){
  start <- (train.num+test.num) %% stp+stp
  mse <- 0
  for (i in seq(start,train.num+test.num-stp,stp)){
    train.mon <- date.lst[1:i]
    test.mon <- date.lst[(i+1):(i+stp)]
    train.data <- tidydata %>% filter(Date %in% train.mon)
    train.data <- as.matrix(train.data[,2:n])
    test.data <- tidydata %>% filter(Date %in% test.mon)
    test.data <- as.matrix(test.data[,2:n])
    
    dtrain <- xgb.DMatrix(train.data[,1:(n-2)],label=train.data[,n-1])
    xgModel <-  xgb.train(data = dtrain, nround = nrounds, objective = "reg:squarederror",params=params)
    
    preds <- predict(xgModel,test.data[,1:(n-2)])
    mse <- mse+sum((test.data[,n-1]-preds)^2)/length(preds)
  }
  return(mse)
}

mse <- rep(0,nrow(xgb_grid))
opt_i <- 0
for (i in 97:(nrow(xgb_grid))){
  params <- list(eta=xgb_grid$eta[i],max_depth=xgb_grid$max_depth[i])
  mse[i] <- ts_cv_mse(params,nrounds=xgb_grid$nrounds[i],4)
  save.image("~/12.15.RData")
}


write.csv(mse,"finalmse.csv")
opt_i <- which.min(mse)
params <- list(eta=xgb_grid$eta[opt_i],max_depth=xgb_grid$max_depth[opt_i])
nrounds <- xgb_grid$nrounds[opt_i]
train.feature <- tidydata %>% select(-Date, -RETMONTH_end) %>% as.matrix()
train.ret <- tidydata %>% select(RETMONTH_end) %>% as.matrix()
dtrain <- xgb.DMatrix(train.feature, label=train.ret)
xgModel <-  xgb.train(data = dtrain, nround = nrounds, objective = "reg:squarederror",params=params)
xgb.save(xgModel, 'return_forecast_model')
#save model


