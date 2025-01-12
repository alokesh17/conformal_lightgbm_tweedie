
library(lightgbm)
library(tidyr)
library(dplyr)
library(tweedie)
library(HDtweedie)
rm(list = ls())
library(lightgbm)
library(tweedie)
library(ggplot2)

## TWEEDIE NEGATIVE LOG-LIKELIHOOD
dtweedie_nlogl <- function(phi, y, mu, power) {
  ## `y` and `mu` MUST BE OF THE SAME LENGTH
  ans <- -2 * mean(log(
    dtweedie(y = y, mu = mu, phi = phi,  power = power)
  ))
  if (is.infinite(ans)) {
    ans <- mean(
      tweedie.dev(y = y, mu = mu, power = power)
    )
  }
  attr(ans, "gradient") <- dtweedie.dldphi(
    y = y, mu = mu, phi = phi, power = power
  )
  return(ans)
}

## LOAD AUTO CLAIM DATA
data(AutoClaim, package = "cplm")

## REMOVE FEATURES
x <- AutoClaim[, -c(1:5, 10, 16, 29)]

## INTEGER CODING TABLE FOR CATEGORICAL FEATURES
coding <- sapply(x, function(x) {
  if(is.factor(x))
    data.frame(
      code = seq_along(levels(x)),
      level = levels(x)
    )
})
coding[sapply(coding, is.null)] <- NULL
print(coding)

## ENCODE CATEGORICAL FEATURES INTO INTEGERS
index <- sapply(x, is.factor)
x[index] <- lapply(x[index], as.integer)

dt <- list(data = as.matrix(x), label = AutoClaim$CLM_AMT5 / 1000)

X=as.matrix(x)
Y=AutoClaim$CLM_AMT5 / 1000
n=length(Y)

alpha=.05
eps=.001
# percentage of 0 is 0.6323
sum(Y==0)/length(Y)
hist(Y)
simulations=100

pow <- seq(1.2, 1.5, by = 0.025)
nrounds <- 2000L
num_leaves <- 10L
learning_rate <- 0.005
nfold <- 5L

asymetric_pearson_residuals_rates=rep(0,simulations)
pearson_residuals_rates=rep(0,simulations)
pearson_residuals_rates_n3=rep(0,simulations)

deviance_residuals_rates=rep(0,simulations)
anscombe_residuals_rates=rep(0,simulations)


asymetric_pearson_residuals_gapOFci=rep(0,simulations)

pearson_residuals_gapOFci_n3=rep(0,simulations)

pearson_residuals_gapOFci=rep(0,simulations)##for test
deviance_residuals_gapOFci=rep(0,simulations)
anscombe_residuals_gapOFci=rep(0,simulations)


best_model_evaluation=function(){
  pow <- seq(1.2, 1.5, by = 0.025)
  nrounds <- 2000L
  num_leaves <- 10L
  learning_rate <- 0.005
  nfold <- 5L
  
  loglik <- rep(NA, length(pow))
  model <- vector("list", length(pow))
  set.seed(2024)
  for (k in seq_along(pow)) {
    pp <- pow[k]
    params <- list(
      num_leaves = num_leaves,
      learning_rate = learning_rate,
      objective = "tweedie",
      tweedie_variance_power = pp,
      verbose = 0L,
      use_missing = TRUE
    )
    
    fit <- lgb.cv(
      params = params,
      data = dtrain,
      nfold = nfold,
      nrounds = nrounds
    )
    best_iter <- fit$best_iter
    # print(best_iter <- fit$best_iter)
    # print(fit$best_score)
    # 
    # ## CROSS-VALIDATION ERROR
    # metric <- unlist(fit$record_evals$valid$tweedie$eval)
    # metric_se <- unlist(fit$record_evals$valid$tweedie$eval_err)
    # 
    # ## U-CURVE FOR CROSS-VALIDATION ERROR
    # data.frame(
    #   metric = metric,
    #   lower = metric - 2 * metric_se,
    #   upper = metric + 2 * metric_se,
    #   boost = rep(1:nrounds, 3)
    # ) |> ggplot() + 
    #   geom_line(aes(x = boost, y = metric), linewidth = 2) +
    #   geom_ribbon(aes(x = boost, ymin = lower, ymax = upper),
    #               alpha = 0.5, fill = "lightblue") +
    #   labs(x = "Number of boosting iterations",
    #        y = "Cross-validation error")
    
    ## REFIT MODEL WITH FULL DATA USING BEST NROUNDS
    fit <- lgb.train(
      params = params,
      data = dtrain,
      nrounds = best_iter
    )
    model[[k]] <- fit
    preds <- predict(fit, dt$data)
    
    ##===== ESTIMATION OF DISPERSION =====##
    
    # ## PEARSON ESTIMATION
    # phi_moment <- mean((dt$label - preds)^2 / (preds^pp))
    # phi <- phi_moment
    
    ## DEVIANCE ESTIMATION
    phi_saddle <- mean(
      tweedie.dev(y = dt$label, mu = preds, power = pp)
    )
    # phi <- phi_saddle
    
    ## MAXIMUM LIKELIHOOD ESTIMATION
    lower_limit <- min(0.001, 0.5 * phi_saddle)
    upper_limit <- 10 * phi_saddle
    ans <- optimize(
      f = dtweedie_nlogl, maximum = FALSE,
      interval = c(lower_limit, upper_limit),
      power = pp, mu = preds, y = dt$label
    )
    phi <- ans$minimum
    
    print(sprintf(
      "Power parameter = %.3f, best iteration = %4d, dispersion parameter = %.4f",
      pp, best_iter, phi
    ))
    loglik[k] <- mean(log(
      dtweedie(y = dt$label, mu = preds, phi = phi, power = pp)
    ))
  }
  
  data.frame(loglik = loglik, power = pow) |>
    ggplot() + geom_line(aes(x = power, y = loglik)) +
    labs(x = "Power parameter", y = "Log-likelihood")
  
  kmax <- which.max(loglik)
  print(pp <- pow[kmax])
  fit <- model[[kmax]]
  
  return(list(fit))
}



for(j in 1:simulations){
  
  n1= 4000
  n2= 4000
  n3= n-n1-n2
  pp=1.25
  
  #samp=sample(1:n1+n2,n1,rep=F)
  ##
  samp=sample(1:(n1+n2),n1,rep=F)
  Xsamp <- X[samp,]
  Ysamp <- Y[samp]
  
  #####
  
  
  train_data <- lgb.Dataset(data = as.matrix(Xsamp),label = Ysamp,
                            categorical_feature = colnames(x)[which(index)])
  
  test_data <- lgb.Dataset(data = as.matrix(X[8001:n,]),label = Y[8001:n],
                           categorical_feature = colnames(x)[which(index)])
  
  train_data_bigger <- lgb.Dataset(data = as.matrix(X[1:8000,]),label = Y[1:8000],
                                   categorical_feature = colnames(x)[which(index)])
  
  
  mod = tweedie.profile(formula=Ysamp~Xsamp,
                        data=data.frame(Ysamp,Xsamp), 
                        p.vec=seq(1.2,1.8,0.1), method='series', do.ci=F)
  
  # Define parameters for LightGBM
  params<- list(
    
    num_leaves = num_leaves,
    learning_rate = learning_rate,
    objective = "tweedie",
    tweedie_variance_power = mod$p.max,
   
    verbose = 0L,
    use_missing = TRUE
    
    
    # objective = "tweedie",
    # #num_class = 3,  # Number of classes in the target variable
    # metric = "tweedie",
    # num_leaves=3,
    # learning_rate=.001,
    # tweedie_variance_power=1.7
  )
  
  # Define parameters for LightGBM alternatively
  params_alt <- list(
    num_leaves = num_leaves,
    learning_rate = learning_rate,
    objective = "l2",
    tweedie_variance_power =  mod$p.max, ##need to get a fixed value
    
    verbose = 0L,
    use_missing = TRUE
    
    
    # objective = "regression",
    # #num_class = 3,  # Number of classes in the target variable
    # metric = "l2",
    # tweedie_variance_power=1.7
  )
  
  
  
  # Train LightGBM model
  model <- lgb.train(params = params,
                     data = train_data,
                     nrounds = 50*j,  # Number of boosting rounds, might overfit
                     verbose = 1
  )
  
  # model_cv<- lgb.cv(params = 
  #                        #params_alt,
  #                        params,
  #                      data = train_data,
  #                      nrounds = 2000,  # Number of boosting rounds, might overfit
  #                      nfold = 3L,
  #                      verbose = 1)
  
  predictions <- predict(model, as.matrix(Xsamp))%>%
    data.frame()
  ###
 
  # feature importance
  # tree_imp = lgb.importance(model, percentage = TRUE)
  # lgb.plot.importance(tree_imp, top_n = 50L, measure = "Gain")
  # 
  
  
  #cvfit_pblpr=cv.glmnet(Xsamp[,-1],Ysamp,family=tweedie(link.power=0,var.power=1.7),alpha=0)##Fit a ridge regression 
  ##?tweedie #a value of q=link power=0 is log(\mu)=x^{'}beta
  Ypred_n1=predict(model,Xsamp
                   #, s=cvfit_pblpr$lambda.min,type = "response" 
  )  
  n2_set=setdiff(1:(n1+n2),samp) ## the n2 set
  Ypred_n2=predict(model,X[n2_set,]
                   #, s=cvfit_pblpr$lambda.min,type = "response" 
  )  
  
  
  
  
  n3_set=c(8001:n)
  Ypred_n3=predict(model,X[n3_set,]
                   #, s=cvfit_pblpr$lambda.min,type = "response" 
  )  
  
  #Obtain Pearson, Deviance, and Anscombe residuals
  pearson_residuals <- (Y[n2_set]-Ypred_n2)/sqrt(7*(Ypred_n2)^1.7)
  deviance_residuals <- 2*(((Y[n2_set])^(2-1.7)/((1-1.7)*(2-1.7)))-(((Y[n2_set])*(Ypred_n2)^(1-1.7))/(1-1.7))+
                             (((Ypred_n2)^(2-1.7))/(2-1.7))  )
  
  #2*(Y[n2_set]*log((Y[n2_set]+eps)/(Ypred_n2+eps))-(Y[n2_set]-Ypred_n2)) #just to avoid NAN
  anscombe_residuals <- sqrt(abs(Y[n2_set]-Ypred_n2))*sign(Y[n2_set]-Ypred_n2)
  
  q=ceiling((1-alpha)*(n2+1))/(n2)
  
  asymmetric_upper_pearson_residuals_quantile=quantile(pearson_residuals,ceiling((1-(.05/2))*(n2+1))/(n2))
  asymmetric_lower_pearson_residuals_quantile=quantile(pearson_residuals,floor(((.05/2))*(n2+1))/(n2))
  
  
  # asymmetric_upper_pearson_residuals_quantile=quantile(pearson_residuals,.975)
  # asymmetric_lower_pearson_residuals_quantile=quantile(pearson_residuals,.025)
  # 
  pearson_residuals_quantile=c(quantile(abs(pearson_residuals),q))
  deviance_residuals_quantile=quantile(deviance_residuals,q)
  anscombe_residuals_quantile=quantile(anscombe_residuals,q)
  
  asymetric_pearson_residuals_CI=cbind((Ypred_n1/sqrt(7*Ypred_n1^1.7)) +asymmetric_lower_pearson_residuals_quantile,
                                       (Ypred_n1/sqrt(7*Ypred_n1^1.7)) +asymmetric_upper_pearson_residuals_quantile)
  
  pearson_residuals_CI=cbind((Ypred_n1/sqrt(7*Ypred_n1^1.7)) -pearson_residuals_quantile,
                             (Ypred_n1/sqrt(7*Ypred_n1^1.7)) +pearson_residuals_quantile)
  
  ##a completely new test set
  pearson_residuals_CI_n3=cbind((Ypred_n3/sqrt(7*Ypred_n3^1.7)) -pearson_residuals_quantile,
                                (Ypred_n3/sqrt(7*Ypred_n3^1.7)) +pearson_residuals_quantile)
  
  
  
  #pearson_residuals_CI=cbind(Ypred_n1 -pearson_residuals_quantile,Ypred_n1 +pearson_residuals_quantile)
  deviance_residuals_CI=cbind(Ypred_n1 -deviance_residuals_quantile,Ypred_n1 +deviance_residuals_quantile)
  anscombe_residuals_CI=cbind(Ypred_n1 -anscombe_residuals,Ypred_n1 + anscombe_residuals)
  
  colnames(asymetric_pearson_residuals_CI)=c("L","U")
  
  colnames(pearson_residuals_CI)=c("L","U")
  colnames(pearson_residuals_CI_n3)=c("L","U")
  
  colnames(deviance_residuals_CI)=c("L","U")
  colnames(anscombe_residuals_CI)=c("L","U")
  
  asymetric_pearson_residuals_rates[j]=asymetric_pearson_residuals_CI%>%as.data.frame()%>%mutate(y=Y[samp])%>%
    rowwise()%>%
    mutate(result=if(((y/sqrt(7*y^1.7 +eps))>L)&&((y/sqrt(7*y^1.7 +eps))<U)){1}else{0})%>%dplyr::select(result)%>%sum()/n1
  
  
  pearson_residuals_rates[j]= pearson_residuals_CI%>%as.data.frame()%>%mutate(y=Y[samp])%>%
    rowwise()%>%
    mutate(result=if(((y/sqrt(7*y^1.7 +eps))>L)&&((y/sqrt(7*y^1.7 +eps))<U)){1}else{0})%>%dplyr::select(result)%>%sum()/n1
  
  
  pearson_residuals_rates_n3[j]= pearson_residuals_CI_n3%>%as.data.frame()%>%mutate(y=Y[8001:n])%>%
    rowwise()%>%
    mutate(result=if(((y/sqrt(7*y^1.7 +eps))>L)&&((y/sqrt(7*y^1.7 +eps))<U)){1}else{0})%>%dplyr::select(result)%>%sum()/n3
  
  # pearson_residuals_rates[j]= pearson_residuals_CI%>%as.data.frame()%>%mutate(y=Y[samp])%>%rowwise()%>%
  #   mutate(result=if((y>L)&&(y<U)){1}else{0})%>%dplyr::select(result)%>%sum()/n1
  # 
  
  asymetric_pearson_residuals_gapOFci[j]=asymetric_pearson_residuals_CI%>%
    as.data.frame()%>%mutate(y=Y[samp])%>%rowwise()%>%
    mutate(diff=U-L)%>%dplyr::select(diff)%>%sum()/n1
  
  pearson_residuals_gapOFci[j]=pearson_residuals_CI%>%as.data.frame()%>%mutate(y=Y[samp])%>%rowwise()%>%
    mutate(diff=U-L)%>%dplyr::select(diff)%>%sum()/n1
  
  pearson_residuals_gapOFci_n3[j]=pearson_residuals_CI_n3%>%as.data.frame()%>%mutate(y=Y[8001:n])%>%rowwise()%>%
    mutate(diff=U-L)%>%dplyr::select(diff)%>%sum()/n3
  
  deviance_residuals_rates[j]= deviance_residuals_CI%>%as.data.frame()%>%mutate(y=Y[samp])%>%rowwise()%>%
    mutate(result=if((y>L)&&(y<U)){1}else{0})%>%dplyr::select(result)%>%sum()/n1
  
  deviance_residuals_gapOFci[j]=deviance_residuals_CI%>%as.data.frame()%>%mutate(y=Y[samp])%>%rowwise()%>%
    mutate(diff=U-L)%>%dplyr::select(diff)%>%sum()/n1
  
  anscombe_residuals_rates[j]= anscombe_residuals_CI%>%as.data.frame()%>%mutate(y=Y[samp])%>%rowwise()%>%
    mutate(result=if((y>L)&&(y<U)){1}else{0})%>%dplyr::select(result)%>%sum()/n1
  
  anscombe_residuals_gapOFci[j]= anscombe_residuals_CI%>%as.data.frame()%>%mutate(y=Y[samp])%>%rowwise()%>%
    mutate(diff=U-L)%>%dplyr::select(diff)%>%sum()/n1
  
  ##
  
  print(j)
}

asymetric_pearson_residuals_rates

pearson_residuals_rates
# deviance_residuals_rates
# anscombe_residuals_rates
asymetric_pearson_residuals_gapOFci
pearson_residuals_gapOFci
pearson_residuals_gapOFci_n3

# deviance_residuals_gapOFci
# anscombe_residuals_gapOFci


plot(1:simulations,pearson_residuals_gapOFci,ty="l",xlab="boosting rounds",col = "red")
lines(1:simulations,pearson_residuals_gapOFci_n3,col="blue")

plot(1:simulations,asymetric_pearson_residuals_gapOFci,ty="l",xlab="boosting rounds")








