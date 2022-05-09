## TO RUN THIS FILE: please set the path according to your working directory.
## In "path", the visuals will be saved.

path = "/home/kim2712/Desktop/research/SENSE/R_vis/figs"
set.seed(1)

library(LearnGeom)
library(plotrix)

n=5000
ITER = 900
vis_interval=20
step_size=0.001

p=0.55




ci=qnorm(.999)
projection = function(x,v){
  #project x=c(1,x1,x2) i.e. (x1,x2) onto v0+v1x1+v2x2=0
  t = -sum(v*x)/(v[2]^2+v[3]^2)
  x1 = v[2]*t + x[2]
  x2 = v[3]*t + x[3]
  return(c(1,x1,x2))
}
projection_mat = function(X,v){
  #project x=c(1,x1,x2) i.e. (x1,x2) onto v0+v1x1+v2x2=0
  t = -colSums(v*t(X))/(v[2]^2+v[3]^2)
  X = outer(t, v, "*") +X
  X[,1]=1
  return(X)
}

#1. Data Generating
lim=8
m=7
sigma=0.2
gamma_=8
epsilon = gamma_*sigma


y=sample(c(1,-1),n, TRUE)
yb = (y+1)/2 # binary form of the same y


z=sample(c(1,-1),n, TRUE, c(p,1-p))
z=z*(y==1)*m
centers = matrix(c(y,z), byrow=FALSE, ncol=2)
Z=rnorm(n*2, mean=0, sd=1)
X = cbind(1, Z*sigma+centers)

h = function(X,b){
  b = matrix(b,ncol=1)
  return(1/(1+exp(-X%*%b)))  
}

pred_y = function(X,y,b){
  b = matrix(b,ncol=1)
  return(1/(1+exp(-(X%*%b)*y)))  
}


perturb = function(b,X,y){
  Py = pred_y(X,y,b)
  #direction = rep(1,length(y))
  #direction[Py>0.5] = -1
  #direction = direction*y
  direction = -y
  X_adv = X +epsilon*direction* matrix(c(0,b[2],b[3])/sqrt(b[2]^2+b[3]^2),ncol=3, nrow= length(y), byrow=T)
  #X_adv = X - epsilon*direction* matrix(c(0,b[2],b[3]),ncol=3, nrow= length(y), byrow=T)
  return(X_adv)
}
#c=0.9
#b = runif(3)
b = adv_model$coefficients

sense = function(b,X,y, c=0.9){
  
  X_adv = perturb(b,X,y)
  
  b_y_Pos=b+c(log((1-c)/c),0,0)
  b_y_Nega=b+c(-log((1-c)/c),0,0)
  
  Proj_y_Pos=projection_mat(X,b_y_Pos)
  Proj_y_Nega=projection_mat(X,b_y_Nega)
  
  Proj_y = Proj_y_Pos
  Proj_y[y<0,] = Proj_y_Nega[y<0,]
  
  Natural_stage = pred_y(X,y,b)<c
  PGD_stage = pred_y(X_adv,y,b)>c
  
  X_sense = X
  X_sense[PGD_stage,] = X_adv[PGD_stage,]
  X_sense[(!Natural_stage)&(!PGD_stage),] = Proj_y[(!Natural_stage)&(!PGD_stage),]
  return(X_sense)
}



par(mfrow=c(1,1))
X1=1
X2 = X3= seq(-lim ,lim, by = 0.1)
grid_set = expand.grid(X2, X3)
colnames(grid_set) = c('X2', 'X3')

background=function(model){
  y_grid = predict(model, data.frame(X1,grid_set), type="response")>0.5
  points(grid_set, pch = 20, col = ifelse(y_grid == 1, 'skyblue','tomato'))  
}


#True class: 
plot(x = X[,2], y = X[,3], xlim=c(-lim,lim), ylim=c(-lim,lim), 
     col=(y>0)+3,main="True class, data",xaxt='n',yaxt='n',pch=".")


par(mfrow=c(1,1))
#2. Visualizing the original data and the logistic regression result

mymodel <- glm(as.factor(y)~X2+X3, data = data.frame(X), family ='binomial')
X_hat = predict(mymodel, data.frame(X), type="response")
plot(X[,2],X[,3], xlim=c(-lim,lim), ylim=c(-lim,lim), 
     col=(y)+3,main="Best logistic linear classifier",xaxt='n',yaxt='n',pch=".",xlab="X1",ylab="X2")
background(mymodel)
points(X[,2],X[,3], col=(y)+3,pch="." )

b0= mymodel$coefficients[1]
b1= mymodel$coefficients[2]
b2= mymodel$coefficients[3]

alpha= -b0/b2 
beta = -b1/b2

abline(a=alpha, b=beta, lwd=7)



common_initial = c(b0,b1,b2) #mymodel$coefficients #runif(3)


##SENSE: 
par(mfrow=c(1,1))
adv_model=mymodel
adv_model$coefficients = common_initial
X_sense = X
for(i in 0:ITER){
  
  
  
  #for model update:
  prob = predict(adv_model, data.frame(X_sense), type="response")
  adv_model$coefficients = adv_model$coefficients + step_size * colSums((yb-prob)*X_sense)
  
  X_hat_by_sense = predict(adv_model, data.frame(X), type="response")
  
  if (i%%vis_interval==0){
    dev.copy(png,paste(path, "/SENSE",i,".png",sep=''))
    
    plot(X[,2],X[,3], xlim=c(-lim,lim), ylim=c(-lim,lim), 
         col=(2*(X_hat_by_adv>0.5)-1)+3,main=paste("SENSE-AT, iter=",i),xaxt='n',yaxt='n',pch=".", xlab="X1",ylab="X2")
    background(adv_model)
    draw.circle(-1,0,ci*sigma+gamma_*sigma,border="white",lwd=2, col="white")
    draw.circle(1,-m,ci*sigma+gamma_*sigma,border="white",lwd=2, col="white")
    draw.circle(1,m,ci*sigma+gamma_*sigma,border="white",lwd=2, col="white")
    draw.circle(-1,0,ci*sigma,border=2,lwd=2)
    draw.circle(1,m,ci*sigma,border=4,lwd=2)
    draw.circle(1,-m,ci*sigma,border=4,lwd=2)
    
    #points(X[,2],X[,3], col=(y)+3,pch="." )
    b0= adv_model$coefficients[1]
    b1= adv_model$coefficients[2]
    b2= adv_model$coefficients[3]
    alpha= -b0/b2 
    beta = -b1/b2
    abline(a=alpha, b=beta, lwd=3, lty=2)
    X_sense = sense(b=adv_model$coefficients,X,y)
    points(X_sense[,2],X_sense[,3], col=(y)+3,pch='*' )
    
    dev.off()
  }
  
}


#3. adversarial example: 

adv_model=mymodel
adv_model$coefficients = common_initial
X_adv = X
par(cex.main=3)
par(cex.lab=1.6)

for(i in 0:ITER){
  
  #for model update:
  prob = predict(adv_model, data.frame(X_adv), type="response")
  adv_model$coefficients = adv_model$coefficients + step_size * colSums((yb-prob)*X_adv)
  
  X_hat_by_adv = predict(adv_model, data.frame(X), type="response")
  
  
  
  if (i%%vis_interval==0){
    dev.copy(png,paste(path, "/R_AT",i,".png",sep=''))
    
    
    plot(X[,2],X[,3], xlim=c(-lim,lim), ylim=c(-lim,lim), 
         col=(2*(X_hat_by_adv>0.5)-1)+3,main=paste("R-AT, iter=",i),xaxt='n',yaxt='n',pch=".", xlab="X1",ylab="X2")
    background(adv_model)
    draw.circle(-1,0,ci*sigma+epsilon,border="white",lwd=2, col="white")
    draw.circle(1,-m,ci*sigma+epsilon,border="white",lwd=2, col="white")
    draw.circle(1,m,ci*sigma+epsilon,border="white",lwd=2, col="white")
    draw.circle(-1,0,ci*sigma,border=2,lwd=2)
    draw.circle(1,m,ci*sigma,border=4,lwd=2)
    draw.circle(1,-m,ci*sigma,border=4,lwd=2)
    
    #points(X[,2],X[,3], col=(y)+3,pch="." )
    b0= adv_model$coefficients[1]
    b1= adv_model$coefficients[2]
    b2= adv_model$coefficients[3]
    alpha= -b0/b2 
    beta = -b1/b2
    abline(a=alpha, b=beta, lwd=3,lty=2)
    X_adv = perturb(b=adv_model$coefficients,X,y)
    points(X_adv[,2],X_adv[,3], col=(y)+3,pch="." )
    dev.off()
  }
}

