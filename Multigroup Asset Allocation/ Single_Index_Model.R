

#Cong Cao 
# single Index model
a1 <- read.table("http://www.stat.ucla.edu/~nchristo/statistics_c183_c283/
stocks5_period1.txt", header=TRUE)
#Regression of r11 on rsp1 (index):
q <- lm(a1$r11 ~ a1$rsp1)
#Summary of the regression above:
summary(q)
#List the names of the results in object q:
names(q)
#Get the estimates of alpha and beta:
q$coefficients[1]
q$coefficients[2]
#List the residuals:
q$residuals
#Get the estimate of the variance of the error term (MSE):
sum(q$residuals^2)/(nrow(a1)-2)
#Another way:
summary(q)$sigma^2
#variance-covariance matrix of the estimates of the main parameters
#of the model:
vcov(q)
#Get the variance of the estimate of beta:
vcov(q)[2,2]
#Another way:
summary(q)$coefficients[4]^2
####################################################################

#Read the data:
data1 <- read.table("http://www.stat.ucla.edu/~nchristo/statistics_c183_c283/
                      stocks_5_ret.txt", header=TRUE)
#Read the data in matrix form:
b <- as.matrix(data1)
#Initialize the vectors and matrices:
x <- rep(0, 30)
xx <- matrix(x, ncol=6, nrow=5)
stock <- rep(0,5)
alpha <- rep(0,5)
beta <- rep(0,5)
mse <- rep(0,5)
Rbar <- rep(0,5)
Ratio <- rep(0,5)
col1 <- rep(0,5)
col2 <- rep(0,5)
col3 <- rep(0,5)
col4 <- rep(0,5)
col5 <- rep(0,5)
#Risk free rate: 
rf <- 0.001
#Perform regression of each stock on the index and record α, β, σ2 :
for(i in 1:5){
  alpha[i] <- lm(data=data1,formula=data1[,i] ~ data1[,6])$coefficients[1]
  beta[i] <- lm(data=data1,formula=data1[,i] ~ data1[,6])$coefficients[2]
  Rbar[i] <- alpha[i]+beta[i]*mean(b[,6])
  mse[i] <- sum(lm(data=data1, formula=data1[,i] ~ data1[,6])$residuals^2)/(nrow(b)-2)
  Ratio[i] <- (Rbar[i]-rf)/beta[i]
  stock[i] <- i
}
#So far we have this table:
xx <- (cbind(stock,alpha, beta, Rbar, mse, Ratio))
#Order the table based on the excess return to beta ratio:
aaa <- xx[order(-Ratio),]

#Create the last 5 columns of the table:
col1 <- (aaa[,4]-rf)*aaa[,3]/aaa[,5]
col3 <- aaa[,3]^2/aaa[,5]
for(i in(1:5)) {
  col2[i] <- sum(col1[1:i])
  col4[i] <- sum(col3[1:i])
}
#So far we have:
cbind(aaa, col1, col2, col3, col4)
#Compute the Ci (col5): 
  for(i in (1:5)) {
  col5[i] <- var(data1[,6])*col2[i]/(1+var(data1[,6])*col4[i])
}

#SHORT SALES ALLOWED:
#Compute the Zi:
z_short <- (aaa[,3]/aaa[,5])*(aaa[,6]-col5[5])
#Compute the xi:
x_short <- z_short/sum(z_short)
#The final table when short sales allowed:
aaaa <- cbind(aaa, col1, col2, col3, col4, col5, z_short, x_short)

#SHORT SALES NOT ALLOWED:
#First create a matrix up to the maximum of col5:
table1 <- cbind(aaa, col1, col2, col3, col4, col5)
table2 <- table1[1:which(col5==max(col5)), ]
#Compute the Zi:
z_no_short <- (table2[,3]/table2[,5])*(table2[,6]-max(col5))
#Compute the xi:
x_no_short <- z_no_short/sum(z_no_short)
#The final table when short sales are not allowed:
aaaaa <- cbind(table2, z_no_short, x_no_short)













