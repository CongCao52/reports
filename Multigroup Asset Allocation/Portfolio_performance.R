# Cong Cao 
#Portfolio performance  

library(stockPortfolio)

#DATA:
ticker <-  c("C", "KEY", "WFC", "JPM",  "SO", "DUK",  "D", "HE",  "EIX",    "AMGN",  "GILD", "CELG",  "BIIB",  "CAT", "DE",  "IMO", "MRO",  "YPF", "^GSPC")

ind <- c("Money Center Banks",  "Money Center Banks",  "Money Center Banks", "Money Center Banks",  "Electrical Utilities", "Electrical Utilities", "Electrical Utilities", "Electrical Utilities", "Electrical Utilities", "Biotechnology",  "Biotechnology", "Biotechnology", "Biotechnology", "Machinery",  "Machinery",  
         "Fuel Refining", "Fuel Refining", "Fuel Refining", "Index")

#Put them together:
data <- as.data.frame(cbind(ticker, ind))

#This is what you have:
data
ticker <- data$ticker
ind <- data$ind

#===> get data <===#
gr1 <- getReturns(ticker, start='1999-12-31', end='2004-12-31')
summary(gr1)
gr1$R		# returns
gr1$ticker  # original ticker
gr1$period  # sample spacing
gr1$start   # when collection started
gr1$end     # when collection ended

#===> model building <===#
#No model (default - Markowitz model):
m1 <- stockModel(gr1, drop=19) # drop index

#Single index model - short sales allowed, default Rf=0.
sim1  <- stockModel(gr1, model='SIM', index= 19)

#Single index model - short sales not allowed (Rf=0):
sim2 <- stockModel(gr1, model='SIM', index= 19, shortSelling=FALSE)


#Constant correlation model - short sales allowed, default Rf=0.:
ccm1  <- stockModel(gr1, model='CCM', drop= 19)

#Constant correlation model - short sales not allowed (Rf=0):
ccm2  <- stockModel(gr1, model='CCM', shortSelling=FALSE,drop= 19)

#Multi group model - short sales allowed, default Rf=0:
mgm1 <- stockModel(gr1, model='MGM', drop= 19, industry=ind)


#===> Optimize <===#
op1  <- optimalPort(m1)

opsim1  <- optimalPort(sim1)
opsim2 <- optimalPort(sim2)

opccm1 <- optimalPort(ccm1)
opccm2 <- optimalPort(ccm2)

opmgm1 <- optimalPort(mgm1)


#Plot:
quartz()
plot(op1)
portPossCurve(m1, add=TRUE)

points(opsim1$risk, opsim1$R, col="green", pch=19)
points(opsim2$risk, opsim2$R, col="blue", pch=19)
points(opccm1$risk, opccm1$R, col="gray", pch=19)
points(opccm2$risk, opccm2$R, col="orange", pch=19)
points(opmgm1$risk, opmgm1$R, col="purple", pch=19)

#How about the equal allocation portfolio:
means <- mean(as.data.frame(gr1$R[,-19]))
var_cov <- cov(gr1$R[,-19])
x_equal <- rep(1,18)/18
Rbar_equal <- t(x_equal) %*% means
sd_equal <- (t(x_equal) %*% var_cov %*% x_equal)^0.5


#Add the equal allocation portfolio on the plot:
points(sd_equal, Rbar_equal, col="dark orange", pch=19)






#===> testing a portfolio on real data <===#
#We will use op1, opsim1, opsim2, opccm1, opccm2, opmgm1 from 2000-2004.
#We will test them on new data set (2005-2009):

#First let's get the data for the testing period:
gr2 <- getReturns(ticker, start='2004-12-31', end='2009-12-31')

options(warn = -1)
tpop1 <- testPort(gr2, op1)
tpopsim1 <- testPort(gr2, opsim1)
tpopsim2 <- testPort(gr2, opsim2)
tpopccm1 <- testPort(gr2, opccm1)
tpopccm2 <- testPort(gr2, opccm2)
tpopmgm1 <- testPort(gr2, opmgm1)

#Also test the equal allocation portfolio:
tpEqu <- testPort(gr2$R[,-19], X=rep(1,18)/18)


#Generate the time plots:
quartz()
plot(tpop1, lty=1, ylim=c(0.4, 3.0))
lines(tpopsim1, lty=2, col="green")
lines(tpopsim2, lty=3, col="blue")
lines(tpopccm1, lty=4, col="yellow")
lines(tpopccm2, lty=5, col="purple")
lines(tpopmgm1, lty=6, col="grey")
lines(tpEqu, lty=7, col="orange")

#Market (S&P500) performance for the same time period:
lines(cumprod(1+rev(gr2$R[, 19])), col="pink", lwd=2)

#Add a legend:
legend('topleft', lty=1:8, c('Markowitz', 'SIM_SS', 'SIM_NSS', 'CCM_SS', 'CCM_NSS', 'MGM_SS', 'EQUAL', 'S&P500'), col=c("black", "green", "blue", "yellow", "purple", "grey", "orange", "pink"))