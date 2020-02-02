data <- read.table("http://www.stat.ucla.edu/~nchristo/statistics_c183_c283/
        returns_5stocks.txt", header=TRUE)
R_ibar <- as.matrix(mean(a))
R_ibar
var_covar <- cov(a)
var_covar
var_covar_inv <- solve(var_covar)
var_covar_inv
Rf <- 0.002
R <- R_ibar-Rf
R
#Compute the vector Z = (Σ^−1)R:
z <- var_covar_inv %*% R
z
x <- z/sum(z)
x
# compute the tangency point G 
R_Gbar <- t(x) %*% R_ibar
R_Gbar
var_G <- t(x) %*% var_covar %*% x
var_G
sd_G <- var_G^0.5
sd_G
slope <- (R_Gbar-Rf)/(sd_G)
slope

# add other points
#(1.3*sd_G,  0.002+slope*(1.3*sd_G))

return_p <- rep(0,10000000);
sd_p <- rep(0,10000000);
j <- 0
i <- 0
for (a in seq(-.2, 1, 0.1)) {
  for (b in seq(-.2, 1, 0.1)) {
    for(c in seq(-.2, 1, 0.1)){
      for(d in seq(-.2, 1, 0.1)){
        for(e in seq(-.2, 1, 0.1)){
          if(a+b+c+d+e==1) {
            j=j+1
            return_p[j]=a*mean(data[,1])+b*mean(data[,2])+
              c*mean(data[,3])+d*mean(data[,4])+e*mean(data[,5])
            sd_p[j]=(a^2*var(data[,1]) +
                       b^2*var(data[,2])+
                       c^2*var(data[,3])+
                       d^2*var(data[,4])+
                       e^2*var(data[,5])+
                       2*a*b*cov(data[,1],data[,2])+
                       2*a*c*cov(data[,1],data[,3])+
                       2*a*d*cov(data[,1],data[,4])+
                       2*a*e*cov(data[,1],data[,5])+
                       2*b*c*cov(data[,2],data[,3])+
                       2*b*d*cov(data[,2],data[,4])+
                       2*b*e*cov(data[,2],data[,5])+
                       2*c*d*cov(data[,3],data[,4])+
                       2*c*e*cov(data[,3],data[,5])+
                       2*d*e*cov(data[,4],data[,5]))^.5
          }
        } } } } }
R_p <- return_p[1:j]
sigma_p <- sd_p[1:j]
# get result from Vectors and Matrices
j <- 0
return_p <- rep(50000)
sd_p <- rep(0,50000)
vect_0 <- rep(0, 50000)
fractions <- matrix(vect_0, 10000,5)
for (a in seq(-.2, 1, 0.1)) {
  for (b in seq(-.2, 1, 0.1)) {
    for(c in seq(-.2, 1, 0.1)){
      for(d in seq(-.2, 1, 0.1)){
        for(e in seq(-.2, 1, 0.1)){
          if(a+b+c+d+e==1) {
            j=j+1
            fractions[j,] <- c(a,b,c,d,e)
            sd_p[j] <-  (t(fractions[j,]) %*% var_covar %*% fractions[j,])^.5
            return_p[j] <-  fractions[j,] %*% R_ibar
          }
        }
      }
    }
  }
}
R_p <- return_p[1:j]
sigma_p <- sd_p[1:j]

plot(sigma_p, R_p,xlab="Risk (standard deviation)", ylab="Expected return",
     xlim=c(0.0,.12), ylim=c(0.0,.016),axes=FALSE, cex=0.4)
axis(1, at=c(0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12))
axis(2,at=c(0,0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016))
lines(c(0,sd_G, 1.3*sd_G),c(.002,R_Gbar,0.002+slope*(1.3*sd_G)))
#Identify portfolio G:
points(sd_G, R_Gbar, cex=2, col="blue", pch=19)
text(sd_G, R_Gbar+.0005, "G")
#Plot the 5 stocks:
points(sd(data), mean(data), pch=19, cex=2.3, col="green")