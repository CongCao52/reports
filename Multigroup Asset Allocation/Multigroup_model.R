
#Cong Cao
#Multi group model 


library(stockPortfolio)

#Stock and index tickers:
ticker <-  c("C", "KEY", "WFC", "SO", "DUK","D","HE", "EIX" ,"AMGN","GILD","CELG","BIIB","IMO",
             "MRO","YPF","^GSPC")


#Industries:
ind <- c("Money Center Banks",  "Money Center Banks",  "Money Center Banks", "Electrical Utilities", "Electrical Utilities", "Electrical Utilities", "Electrical Utilities", "Electrical Utilities",  "Biotechnology", "Biotechnology", "Biotechnology", "Biotechnology", "Fuel Refining", "Fuel Refining", "Fuel Refining", "Index")

#Put them together:
data <- as.data.frame(cbind(ticker, ind))

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


#Multi group model (short sales, Rf=0):
mc  <- stockModel(gr1, model='MGM', drop=16, industry=ind)

#===> identify the optimal portfolio for multi group model<===#
opmc <- optimalPort(mc)

#Plot the optimal portfolio and the stocks:
plot(opmc, xlim=c(0,0.25))


#Add the portfolio possibilities curve:
portPossCurve(mc, add=TRUE, riskRange=5)


#Draw the tangent line:
slope <- (opmc$R-0)/opmc$risk

segments(0,0,2*opmc$risk, mc$Rf+slope*2*opmc$risk)