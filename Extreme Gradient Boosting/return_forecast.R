library(tidyverse)
library(xgboost)
library(lubridate)

return_forecast <- function(datacsv){
  
  ## data preprocession ##
  ########################################
  data <- read.csv(datacsv)
  end.col <- which(str_detect(colnames(data), ".end")) #find the column of end_of_month data
  tidydata <- data %>%  
    arrange(compid, Date) %>%  #order data by company id & Date
    group_by(compid) %>%  #group dataframe by different company id
    mutate_at(.funs = list(adj = ~dplyr::lag(., n = 1, default = NA)),
              .vars = end.col) %>% #shift data within each group (company)
    ungroup() %>% 
    mutate(Industry = as.numeric(factor(Industry)),  #transform the Industry feature to numerical data
           mon = month(as.Date(Date))) %>%  #add the month feature
    select(-end.col, RETMONTH_end, -RETMONTH_end_adj) %>% 
    drop_na() #drop rows with missing value
  date_compid = tidydata[,1:2] # date and company id
  data.x = tidydata[,3:(dim(tidydata)[2]-1)] # features
  ########################################
  
  
  
  ## loading trained model ##
  ########################################
  model = xgb.load('return_forecast_model')
  ########################################
  
  
  
  
  ## prediction: output a dataframe(called pred_result) with Date, compid, predicted label(1:grow,0:fall) ##
  ########################################
  pred_return = predict(model, as.matrix(data.x))
  pred_result = cbind(date_compid, pred_return)
  return(pred_result)
  ########################################
  
}


pred_result = return_forecast("mf850-finalproject-data.csv")
