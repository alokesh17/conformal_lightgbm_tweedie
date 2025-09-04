rm(list = ls())
library(lightgbm)
library(tweedie)
library(foreach)
library(statmod)
library(doParallel)
library(glmnet)

j <- sim_id
cat("Starting replicate:", j, "\n")

set.seed(123)

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

root_search <- function(pred, thresh, p) {
  a <- pred^(1 - p) / (p - 1)
  b <- 1 / (p - 1) / (2 - p)
  c <- thresh^2 - pred^(2 - p) / (2 - p)
  f <- function(y) {
    a * y - b * y^(2 - p) - c
  }
  if (c > 0) {
    lower <- (b / a)^(1 / (p - 1))
    upper <- 10 * lower
  } else {
    lower <- 0
    upper <- (b / a)^(1 / (p - 1))
  }
  roots <- rootSolve::uniroot.all(f, lower = lower, upper = upper)
  root <- roots[length(roots)]
  if (length(roots) < 1L) root <- ymax
  return(root)
}

allocated_cores <- parallel::detectCores() - 1

cat("Using", allocated_cores, "cores\n")
cl <- makeCluster(allocated_cores)
registerDoParallel(cl)

# best model eval function w/ parallel loops ------

best_model_evaluation <- function(dt) {
  ## 1) prepare LGB data & power grid
  dtrain <- lgb.Dataset(
    data                = as.matrix(dt$data),
    label               = dt$label,
    categorical_feature = dt$categorical_feature
  )
  pow <- seq(1.1, 1.9, by = 0.1)
  nP  <- length(pow)
  
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
  
  num_leaves    <- 10
  learning_rate <- 0.005
  nfold         <- 5L
  nrounds       <- 2000L
  
  ## 2) run the grid in parallel
  results <- foreach(k = seq_along(pow),
                     .packages = c("lightgbm","statmod",
                                   "tweedie")) %dopar% {
                                     pp <- pow[k]
                                     
                                     
                                     
                                     params <- list(
                                       num_leaves             = num_leaves,
                                       learning_rate          = learning_rate,
                                       objective              = "tweedie",
                                       tweedie_variance_power = pp,
                                       verbose                = 0L,
                                       use_missing            = TRUE,
                                       num_threads            = 1
                                     )
                                     
                                     ## a) CV to pick # trees
                                     cvm <- lgb.cv(
                                       params  = params,
                                       data    = dtrain,
                                       nfold   = nfold,
                                       nrounds = nrounds
                                     )
                                     best_iter_k <- cvm$best_iter
                                     
                                     ## b) retrain on full data
                                     fit_k <- lgb.train(
                                       params  = params,
                                       data    = dtrain,
                                       nrounds = best_iter_k
                                     )
                                     mu_k <- predict(fit_k, as.matrix(dt$data))
                                     
                                     ## c) estimate phi via MLE
                                     phi_saddle_k <- mean(tweedie.dev(dt$label, mu_k, pp))
                                     low  <- min(0.001, 0.5 * phi_saddle_k)
                                     high <- 10 * phi_saddle_k
                                     opt  <- optimize(
                                       f        = dtweedie_nlogl,
                                       interval = c(low, high),
                                       maximum  = FALSE,
                                       power    = pp,
                                       mu       = mu_k,
                                       y        = dt$label
                                     )
                                     phi_k <- opt$minimum
                                     
                                     ## d) compute profile log-lik
                                     llk <- mean(log(dtweedie(y = dt$label,
                                                              mu = mu_k,
                                                              phi = phi_k,
                                                              power = pp)))
                                     
                                     ## return a list of results for this k
                                     list(
                                       power     = pp,
                                       best_iter = best_iter_k,
                                       phi       = phi_k,
                                       loglik    = llk
                                     )
                                   }
  
  ## 3) unpack results
  best_iter_vec <- sapply(results, `[[`, "best_iter")
  disp_vec      <- sapply(results, `[[`, "phi")
  loglik_vec    <- sapply(results, `[[`, "loglik")
  power_vec     <- sapply(results, `[[`, "power")
  
  ## 4) pick the best power by highest log-lik
  kmax <- which.max(loglik_vec)
  
  print(
    sprintf(
      "Best power parameter = %.3f and dispersion = %.4f",
      power_vec[kmax], disp_vec[kmax]
    )
  )
  
  return(list(
    power     = power_vec[kmax],
    dispersion= disp_vec[kmax],
    best_iter = best_iter_vec[kmax]
  ))
}

## LOAD AUTO CLAIM DATA
data(AutoClaim, package = "cplm")

## REMOVE FEATURES

x <- AutoClaim[, -c(1:5,10,16, 29)]
x$SAMEHOME <- abs(x$SAMEHOME)
which(sapply(x, function(x) any(is.na(x))))

which(sapply(AutoClaim[, -c(1:5,10,16, 29)],
             function(x) any(is.na(x))))


## ENCODE CATEGORICAL FEATURES INTO INTEGERS
index <- sapply(x, is.factor)
x[index] <- lapply(x[index], as.integer)
categorical_feature <- colnames(x)[which(index)]
y <- AutoClaim$CLM_AMT5 / 1000

ymax <- max(y)
n <- nrow(AutoClaim)
n1 <- 4000L
n2 <- 4000L
n3 <- n - n1 - n2
alpha <- 0.05
qq <- ceiling((1 - alpha) * (n2 + 1))
qqL <- floor((alpha/2)*(n2+1))
qqU <- ceiling((1-alpha/2)*(n2+1))

nsim <- 100L
maxit <- 1500L
stepsize <- 25L
steps <- seq(stepsize, maxit, by = stepsize)
nsteps <- length(steps)

pearson_abs_rates   <- numeric(1)
deviance_abs_rates  <- numeric(1)
anscombe_abs_rates  <- numeric(1)
regular_abs_rates <- numeric(1)
lw_abs_rates <- numeric(1)
lwp_abs_rates <- numeric(1)

pearson_abs_widths  <- numeric(1)
deviance_abs_widths <- numeric(1)
anscombe_abs_widths <- numeric(1)
regular_abs_widths <- numeric(1)
lw_abs_widths <- numeric(1)
lwp_abs_widths <- numeric(1)

pearson_raw_rates   <- numeric(1)
deviance_raw_rates  <- numeric(1)
anscombe_raw_rates  <- numeric(1)
regular_raw_rates  <- numeric(1)
lw_raw_rates <- numeric(1)
lwp_raw_rates <- numeric(1)

pearson_raw_widths  <- numeric(1)
deviance_raw_widths <- numeric(1)
anscombe_raw_widths <- numeric(1)
regular_raw_widths <- numeric(1)
lw_raw_widths <- numeric(1)
lwp_raw_widths <- numeric(1)

predsD3 <- numeric(n3)
intervalsD3_pearson_abs <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_deviance_abs <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_anscombe_abs <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_regular_abs <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_lw_abs <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_lwp_abs <- matrix(data=NA, nrow = n3, ncol = 2)

intervalsD3_pearson_raw <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_deviance_raw <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_anscombe_raw <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_regular_raw <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_lw_raw <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_lwp_raw <- matrix(data=NA, nrow = n3, ncol = 2)

## LIGHTGBM PARAMETERS
nrounds <- 2000L
num_leaves <- 10L
learning_rate <- 0.005
nfold <- 5L


cat("Running simulation round", j, "...\n")  
shuffle_index <- sample(n)
dtrain <- list(
  data = as.matrix(x[shuffle_index[1L:n1], ]),
  label = y[shuffle_index[1L:n1]],
  categorical_feature = categorical_feature
)
dtest <- list(
  data = as.matrix(x[shuffle_index[n1 + (1L:n2)], ]),
  label = y[shuffle_index[n1 + (1L:n2)]], 
  categorical_feature = categorical_feature
)
dholdout <- list(
  data = as.matrix(x[shuffle_index[n1 + n2 + (1L:n3)], ]),
  label = y[shuffle_index[n1 + n2 + (1L:n3)]], 
  categorical_feature = categorical_feature
)

## TRAIN MODEL ON PROPER TRAINING SET -----
system.time(fit_info <- best_model_evaluation(dtrain))

pp         <- fit_info$power
phi        <- fit_info$dispersion
best_iter  <- fit_info$best_iter

## REFIT MODEL WITH VARYING NUMBER OF TREES
params <- list(
  num_leaves = num_leaves,
  learning_rate = learning_rate,
  objective = "tweedie",
  tweedie_variance_power = pp,
  verbose = 0L,
  use_missing = TRUE,
  num_threads            = 1
)

system.time(model <- lgb.train(
  params = params,
  data = lgb.Dataset(
    data = as.matrix(dtrain$data),
    label = dtrain$label,
    categorical_feature = categorical_feature
  ),
  nrounds = best_iter
))




## CALCULATE CONFORMITY SCORES ON CALIBRATION SET
preds <- predict(
  object = model,
  newdata = as.matrix(dtest$data),
  num_iteration = best_iter
)
raw_residuals <- dtest$label - preds
pearson_residuals <- (
  dtest$label - preds
) / (preds^(pp/2))
deviance_residuals <- sqrt(
  tweedie.dev(dtest$label, preds, pp)
) * sign(dtest$label - preds)
anscombe_residuals <- (
  dtest$label^(1 - pp/3) - preds^(1 - pp/3)
) / (preds^(pp/6))
regular_residuals <- dtest$label - preds

preds_mad <- predict(
  object = model,
  newdata = as.matrix(dtrain$data),
  num_iteration = best_iter
)

dtrain_mad <- list(
  data = as.matrix(x[shuffle_index[1L:n1], ]),
  label = abs(y[shuffle_index[1L:n1]]-preds_mad),
  categorical_feature = categorical_feature
)

dtrain_pearson <- list(
  data = as.matrix(x[shuffle_index[1L:n1], ]),
  label = abs(y[shuffle_index[1L:n1]]-preds_mad)/ (preds_mad^(pp/2)),
  categorical_feature = categorical_feature
)


# MAD model ----

params_mad <- list(
  num_leaves = num_leaves,
  learning_rate = learning_rate,
  objective = "regression_l2",
  verbose = 0L,
  use_missing = TRUE,
  num_threads            = 1
)

system.time(cvm_mad <- lgb.cv(
  params  = params_mad,
  data    = lgb.Dataset(
    data = as.matrix(dtrain_mad$data),
    label = dtrain_mad$label,
    categorical_feature = categorical_feature
  ),
  nfold   = nfold,
  nrounds = nrounds
))

system.time(cvm_pearson <- lgb.cv(
  params  = params_mad,
  data    = lgb.Dataset(
    data = as.matrix(dtrain_pearson$data),
    label = dtrain_pearson$label,
    categorical_feature = categorical_feature
  ),
  nfold   = nfold,
  nrounds = nrounds
))


system.time(model_mad <- lgb.train(
  params  = params_mad,
  data    = lgb.Dataset(
    data = as.matrix(dtrain_mad$data),
    label = dtrain_mad$label,
    categorical_feature = categorical_feature
  ),
  nrounds = cvm_mad$best_iter
))

system.time(model_pearson <- lgb.train(
  params  = params_mad,
  data    = lgb.Dataset(
    data = as.matrix(dtrain_pearson$data),
    label = dtrain_pearson$label,
    categorical_feature = categorical_feature
  ),
  nrounds = cvm_pearson$best_iter
))

preds_mad <- predict(
  object = model_mad,
  newdata = as.matrix(dtest$data),
  num_iteration = cvm_mad$best_iter
)

preds_pearson <- predict(
  object = model_pearson,
  newdata = as.matrix(dtest$data),
  num_iteration = cvm_pearson$best_iter
)

lw_residuals <- (dtest$label-preds)/preds_mad

lwp_residuals <- ((dtest$label - preds)/(preds^(pp/2)))/preds_pearson


## CALCULATE THRESHOLDS OF CALIBRATION SET RESIDUALS
pearson_abs_quantile <- sort(abs(pearson_residuals))[qq]
deviance_abs_quantile <- sort(abs(deviance_residuals))[qq]
anscombe_abs_quantile <- sort(abs(anscombe_residuals))[qq]
regular_abs_quantile <- sort(abs(regular_residuals))[qq]
lw_abs_quantile <- sort(abs(lw_residuals))[qq]
lwp_abs_quantile <- sort(abs(lwp_residuals))[qq]



pearson_raw_quantile <- c(sort((pearson_residuals))[qqL],
                          sort((pearson_residuals))[qqU])
deviance_raw_quantile <- c(sort((deviance_residuals))[qqL],
                           sort((deviance_residuals))[qqU])
anscombe_raw_quantile <- c(sort((anscombe_residuals))[qqL],
                           sort((anscombe_residuals))[qqU])
regular_raw_quantile <- c(sort((regular_residuals))[qqL],
                          sort((regular_residuals))[qqU])
lw_raw_quantile <- c(sort((lw_residuals))[qqL],
                     sort((lw_residuals))[qqU])
lwp_raw_quantile <- c(sort((lwp_residuals))[qqL],
                      sort((lwp_residuals))[qqU])

## PREDICTIONS ON HOLDOUT SET ----
preds <- predict(
  object = model,
  newdata = as.matrix(dholdout$data),
  num_iteration = best_iter
)
preds_mad <- predict(
  object = model_mad,
  newdata = as.matrix(dholdout$data),
  num_iteration = cvm_mad$best_iter
)
preds_pearson <- predict(
  object = model_pearson,
  newdata = as.matrix(dholdout$data),
  num_iteration = cvm_pearson$best_iter
)
raw_residuals_ho <- dholdout$label - preds
pearson_residuals_ho <- (
  dholdout$label - preds
) / (preds^(pp/2))
deviance_residuals_ho <- sqrt(
  tweedie.dev(dholdout$label, preds, pp)
) * sign(dholdout$label - preds)
anscombe_residuals_ho <- (
  dholdout$label^(1 - pp/3) - preds^(1 - pp/3)
) / (preds^(pp/6))
regular_residuals_ho <- dholdout$label - preds
lw_residuals_ho <- (dholdout$label-preds)/preds_mad
lwp_residuals_ho <- ((dholdout$label - preds)/(preds^(pp/2)))/preds_pearson

## COVERAGE RATE ----
pearson_abs_rates <- mean(
  abs(pearson_residuals_ho) <= pearson_abs_quantile
)
deviance_abs_rates <- mean(
  abs(deviance_residuals_ho) <= deviance_abs_quantile
)
anscombe_abs_rates <- mean(
  abs(anscombe_residuals_ho) <= anscombe_abs_quantile
)
regular_abs_rates <- mean(
  abs(regular_residuals_ho) <= regular_abs_quantile
)
lw_abs_rates <- mean(
  abs(lw_residuals_ho) <= lw_abs_quantile
)
lwp_abs_rates <- mean(
  abs(lwp_residuals_ho) <= lwp_abs_quantile
)

pearson_raw_rates <- mean(
  (pearson_residuals_ho <= pearson_raw_quantile[2]) &
    (pearson_residuals_ho >= pearson_raw_quantile[1])
)
deviance_raw_rates <- mean(
  (deviance_residuals_ho <= deviance_raw_quantile[2]) &
    (deviance_residuals_ho >= deviance_raw_quantile[1])
)
anscombe_raw_rates <- mean(
  (anscombe_residuals_ho <= anscombe_raw_quantile[2]) &
    (anscombe_residuals_ho >= anscombe_raw_quantile[1])
)
regular_raw_rates <- mean(
  (regular_residuals_ho <= regular_raw_quantile[2]) &
    (regular_residuals_ho >= regular_raw_quantile[1])
)
lw_raw_rates <- mean(
  (lw_residuals_ho <= lw_raw_quantile[2]) &
    (lw_residuals_ho >= lw_raw_quantile[1])
)
lwp_raw_rates <- mean(
  (lwp_residuals_ho <= lwp_raw_quantile[2]) &
    (lwp_residuals_ho >= lwp_raw_quantile[1])
)

## LENGTH OF PREDICTION INTERVAL
pearson_abs_widths <-   mean(
  (preds + preds^(pp/2) * pearson_abs_quantile) -
    pmax(0, preds - preds^(pp/2) * pearson_abs_quantile)
)
regular_abs_widths <- mean((preds+regular_abs_quantile)
                           -pmax(0,preds-regular_abs_quantile))
lw_abs_widths <- mean((preds+preds_mad*lw_abs_quantile)-
                        pmax(0,preds-preds_mad*lw_abs_quantile))
lwp_abs_widths <- mean((preds+preds_pearson*lwp_abs_quantile*preds^(pp/2))-
                         pmax(0,preds-
                                preds_pearson*lwp_abs_quantile*preds^(pp/2)))

pearson_raw_widths <- mean(
  pmax(0,preds + preds^(pp/2) * pearson_raw_quantile[2]) -
    pmax(0, preds + preds^(pp/2) * pearson_raw_quantile[1])
)
regular_raw_widths <- mean(pmax(0,preds+regular_raw_quantile[2])
                           -pmax(0,preds+regular_raw_quantile[1]))
lw_raw_widths <- mean(pmax(0,preds+preds_mad*lw_raw_quantile[2])-
                        pmax(0,preds+preds_mad*lw_raw_quantile[1]))
lwp_raw_widths <- 
  mean(pmax(0,preds+preds_pearson*lwp_raw_quantile[2]*preds^(pp/2))-
         pmax(0,preds+preds_pearson*lwp_raw_quantile[1]*preds^(pp/2)))

anscombe_abs_widths <- mean(
  (
    preds^(1 - pp/3) + 
      anscombe_abs_quantile * preds^(pp/6)
  )^(3/(3 - pp)) - (
    pmax(0, preds^(1 - pp/3) - 
           anscombe_abs_quantile * preds^(pp/6))
  )^(3/(3 - pp))
)

anscombe_raw_widths <- mean(
  (
    pmax(0,preds^(1 - pp/3) + 
           anscombe_raw_quantile[2] * preds^(pp/6))
  )^(3/(3 - pp)) - (
    pmax(0, preds^(1 - pp/3) + 
           anscombe_raw_quantile[1] * preds^(pp/6))
  )^(3/(3 - pp))
)  

deviance_abs_widths <- mean(
  sapply(preds, root_search, thresh = deviance_abs_quantile, p = pp)
)

# abs resid intervals lightGBM ----

intervalsD3_pearson_abs<- cbind(
  pmax(0, preds - preds^(pp/2) * pearson_abs_quantile),
  (preds + preds^(pp/2) * pearson_abs_quantile) 
  
)

intervalsD3_deviance_abs <- cbind(0,
                                  sapply(
                                    preds, root_search, 
                                    thresh = deviance_abs_quantile, p = pp))

intervalsD3_anscombe_abs <-  cbind(
  (
    preds^(1 - pp/3) + 
      anscombe_abs_quantile * preds^(pp/6)
  )^(3/(3 - pp)) , (
    pmax(0, preds^(1 - pp/3) - 
           anscombe_abs_quantile * preds^(pp/6))
  )^(3/(3 - pp))
)[,c(2,1)]

intervalsD3_regular_abs<- cbind((preds+regular_abs_quantile)
                                ,pmax(0,preds-regular_abs_quantile))[,c(2,1)]

intervalsD3_lwp_abs<- cbind((preds+preds_pearson*lwp_abs_quantile*preds^(pp/2)),
                            pmax(0,
                                 preds-
                                   preds_pearson*lwp_abs_quantile*preds^(pp/2)))[,c(2,1)] 


intervalsD3_lw_abs<- cbind((preds+preds_mad*lw_abs_quantile),
                           pmax(0,preds-preds_mad*lw_abs_quantile))[,c(2,1)]

# raw resid intervals lightGBM ----
intervalsD3_pearson_raw<- cbind(
  pmax(0,preds + preds^(pp/2) * pearson_raw_quantile[2]) ,
  pmax(0, preds + preds^(pp/2) * pearson_raw_quantile[1])
)[,c(2,1)]

# intervalsD3_deviance_raw<-
intervalsD3_anscombe_raw<- cbind(
  (
    pmax(0,preds^(1 - pp/3) + 
           anscombe_raw_quantile[2] * preds^(pp/6))
  )^(3/(3 - pp)) , (
    pmax(0, preds^(1 - pp/3) + 
           anscombe_raw_quantile[1] * preds^(pp/6))
  )^(3/(3 - pp))
)[,c(2,1)]

intervalsD3_regular_raw<-cbind(pmax(0,preds+regular_raw_quantile[2])
                               ,pmax(0,preds+regular_raw_quantile[1]))[,c(2,1)]
intervalsD3_lw_raw<-cbind(pmax(0,preds+preds_mad*lw_raw_quantile[2]),
                          pmax(0,preds+preds_mad*lw_raw_quantile[1]))[,c(2,1)]
intervalsD3_lwp_raw<- cbind(
  pmax(0,preds+preds_pearson*lwp_raw_quantile[2]*preds^(pp/2)),
  pmax(0,preds+preds_pearson*lwp_raw_quantile[1]*preds^(pp/2)))[,c(2,1)]


# List of all interval matrices ----
interval_list <- list(
  pearson_raw = intervalsD3_pearson_raw,
  deviance_raw= intervalsD3_deviance_raw,
  anscombe_raw = intervalsD3_anscombe_raw,
  regular_raw = intervalsD3_regular_raw,
  lw_raw = intervalsD3_lw_raw,
  lwp_raw = intervalsD3_lwp_raw
)

# Check dimensions
sapply(interval_list, dim)

# Check heads (first 6 rows of each)
lapply(interval_list, head)


responseD3 <- dholdout$label
covariatesD3 <- dholdout$data
predsD3 <- preds

# saving lightGBM ----

save(model,
     pp,
     phi,
     best_iter,
     shuffle_index,
     
     model_mad,
     model_pearson,
     cvm_mad,
     cvm_pearson,
     pearson_abs_rates, deviance_abs_rates, anscombe_abs_rates,
     pearson_abs_widths, deviance_abs_widths, anscombe_abs_widths,
     pearson_raw_rates, deviance_raw_rates, anscombe_raw_rates,
     pearson_raw_widths, deviance_raw_widths, anscombe_raw_widths,
     regular_abs_rates,
     regular_raw_rates,
     regular_abs_widths,
     regular_raw_widths,
     lw_abs_rates,
     lw_raw_rates,
     lw_abs_widths,
     lw_raw_widths,
     lwp_abs_rates,
     lwp_raw_rates,
     lwp_abs_widths,
     lwp_raw_widths,
     
     regular_residuals_ho,
     pearson_residuals_ho,
     anscombe_residuals_ho,
     deviance_residuals_ho,
     lw_residuals_ho,
     lwp_residuals_ho,
     
     responseD3,
     predsD3,
     
     
     intervalsD3_pearson_abs,
     intervalsD3_deviance_abs,
     intervalsD3_anscombe_abs,
     intervalsD3_regular_abs,
     intervalsD3_lw_abs,
     intervalsD3_lwp_abs,
     
     intervalsD3_pearson_raw,
     intervalsD3_deviance_raw,
     intervalsD3_anscombe_raw,
     intervalsD3_regular_raw,
     intervalsD3_lw_raw,
     intervalsD3_lwp_raw,
     
     pearson_abs_quantile,
     deviance_abs_quantile,
     anscombe_abs_quantile,
     regular_abs_quantile,
     lw_abs_quantile,
     lwp_abs_quantile,
     
     pearson_raw_quantile,
     deviance_raw_quantile,
     anscombe_raw_quantile,
     regular_raw_quantile,
     lw_raw_quantile,
     lwp_raw_quantile,
     
     pearson_residuals,
     deviance_residuals,
     anscombe_residuals,
     regular_residuals,
     lw_residuals,
     lwp_residuals,
     
     
     
     file=paste0("sim_round_gbm",j,".rda"))

cat(sprintf("Best Tweedie Power: %.1f\n", pp))
cat(sprintf("Best Tweedie Dispersion: %f\n", phi))
## Clean up  stop cluster -----
stopCluster(cl)
cat("Finished replicate", j, "\n")

##— Load & preprocess data —##
data(AutoClaim, package = "cplm")
dt <- AutoClaim[, -c(1:3, 5, 29, 16, 10)]
dt$CLM_AMT5 <- dt$CLM_AMT5 / 1000
dt$YOJ[is.na(dt$YOJ)]     <- mean(dt$YOJ,     na.rm = TRUE)
dt$INCOME[is.na(dt$INCOME)] <- median(dt$INCOME, na.rm = TRUE)
dt$HOME_VAL[is.na(dt$HOME_VAL)] <- median(dt$HOME_VAL, na.rm = TRUE)
dt$SAMEHOME[is.na(dt$SAMEHOME)] <- mean(dt$SAMEHOME, na.rm = TRUE)
dt$SAMEHOME <- abs(dt$SAMEHOME)

##— Simulation parameters —##
n    <- nrow(dt)
n1   <- 4000L
n2   <- 4000L
alpha <- 0.05
qq   <- ceiling((1 - alpha) * (n2 + 1))
qqL  <- floor((alpha/2) * (n2 + 1))
qqU  <- ceiling((1 - alpha/2) * (n2 + 1))

nsim             <- 100L
alpha_glmnet     <- seq(0, 1, 0.1)
tweedie_power_seq <- seq(1.1, 1.9, by = 0.1)
NRitr            <- 1000

allocated_cores <- parallel::detectCores() - 1

cat("Using", allocated_cores, "cores\n")
cl <- makeCluster(allocated_cores)
registerDoParallel(cl)

# Prepare storage for this replicate ----
pearson_abs_rates   <- numeric(1)
deviance_abs_rates  <- numeric(1)
anscombe_abs_rates  <- numeric(1)
regular_abs_rates <- numeric(1)

pearson_abs_widths  <- numeric(1)
deviance_abs_widths <- numeric(1)
anscombe_abs_widths <- numeric(1)
regular_abs_widths <- numeric(1)


pearson_raw_rates   <- numeric(1)
deviance_raw_rates  <- numeric(1)
anscombe_raw_rates  <- numeric(1)
regular_raw_rates  <- numeric(1)


pearson_raw_widths  <- numeric(1)
deviance_raw_widths <- numeric(1)
anscombe_raw_widths <- numeric(1)
regular_raw_widths <- numeric(1)


simPI_rates         <- numeric(1)
simPI_widths        <- numeric(1)


intervalsD3_pearson_abs <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_deviance_abs <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_anscombe_abs <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_regular_abs <- matrix(data=NA, nrow = n3, ncol = 2)


intervalsD3_pearson_raw <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_deviance_raw <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_anscombe_raw <- matrix(data=NA, nrow = n3, ncol = 2)
intervalsD3_regular_raw <- matrix(data=NA, nrow = n3, ncol = 2)


param_grid <- expand.grid(
  k1 = seq_along(alpha_glmnet),
  k2 = seq_along(tweedie_power_seq)
)

## Run just one iteration (replicate j) ##
cat("Running simulation round", j, "...\n")
# shuffle_index <- sample(n)
dtrain   <- dt[shuffle_index[1:n1], ]
dtest    <- dt[shuffle_index[(n1 + 1):(n1 + n2)], ]
dholdout <- dt[shuffle_index[(n1 + n2 + 1):n], ]

dtrain.test <- rbind(dtrain,dtest)

## Calibration fold ids ##
dtrain_numeric <- model.matrix(~ ., data = dtrain)
foldid <- sample(1:10, size = nrow(dtrain_numeric), replace = TRUE)

## Inner grid search in parallel ##
results <- foreach(i = seq_len(nrow(param_grid)),
                   .packages = c("glmnet", "statmod")) %dopar% {
                     k1 <- param_grid$k1[i]
                     k2 <- param_grid$k2[i]
                     alpha_val     <- alpha_glmnet[k1]
                     tweedie_power <- tweedie_power_seq[k2]
                     
                     glmnet.control(mxitnr = NRitr)
                     cv_model <-cv.glmnet(
                       x        = as.matrix(dtrain_numeric[, -c(1,2)]),
                       y        = as.numeric(dtrain$CLM_AMT5),
                       family   = tweedie(link.power = 0, var.power = tweedie_power),
                       alpha    = alpha_val,
                       parallel = TRUE,
                       foldid   = foldid
                     )
                     
                     # Return a list with the indices and the model
                     list(k1 = k1, k2 = k2, model = cv_model)
                   }


cv_models <- vector("list", length(alpha_glmnet))
for (res in results) {
  if (is.null(cv_models[[res$k1]])) {
    cv_models[[res$k1]] <- vector("list", length(tweedie_power_seq))
    names(cv_models[[res$k1]]) <- paste0("tweedie_", tweedie_power_seq)
  }
  cv_models[[res$k1]][[res$k2]] <- res$model
}

cv_errors <- matrix(NA, nrow = length(alpha_glmnet), ncol = length(tweedie_power_seq))
rownames(cv_errors) <- paste0("alpha_", alpha_glmnet)
colnames(cv_errors) <- paste0("tweedie_", tweedie_power_seq)

# Loop through each cell in the cv_models list and extract the min CV error
for (i1 in seq_along(cv_models)) {
  for (j1 in seq_along(cv_models[[i1]])) {
    cv_errors[i1, j1] <- min(cv_models[[i1]][[j1]]$cvm)
  }
}

# Find the best combination of alpha and tweedie power
best_indices <- which(cv_errors == min(cv_errors, na.rm = TRUE), arr.ind = TRUE)
best_alpha <- alpha_glmnet[best_indices[1]]
best_tweedie_power <- tweedie_power_seq[best_indices[2]]
best_cv_model <- cv_models[[best_indices[1]]][[best_indices[2]]]
best_lambda <- best_cv_model$lambda.min



## CALCULATE CONFORMITY SCORES ON CALIBRATION SET
dtest_numeric <- model.matrix(~ . , data = dtest)

tweedie_power <- best_tweedie_power

preds <- predict(best_cv_model,
                 newx = as.matrix(dtest_numeric[,-c(1,2)]),
                 s = best_lambda,
                 type = "response") 

pearson_residuals <- (
  dtest$CLM_AMT5 - preds
) / (preds^(tweedie_power/2))
deviance_residuals <- sqrt(
  tweedie.dev(dtest$CLM_AMT5, preds, tweedie_power)
) * sign(dtest$CLM_AMT5 - preds)
anscombe_residuals <- (
  dtest$CLM_AMT5^(1 - tweedie_power/3) - preds^(1 - tweedie_power/3)
) / (preds^(tweedie_power/6))
regular_residuals <- dtest$CLM_AMT5 - preds


## CALCULATE THRESHOLDS OF CALIBRATION SET RESIDUALS
pearson_abs_quantile <- sort(abs(pearson_residuals))[qq]
deviance_abs_quantile <- sort(abs(deviance_residuals))[qq]
anscombe_abs_quantile <- sort(abs(anscombe_residuals))[qq]
regular_abs_quantile <- sort(abs(regular_residuals))[qq]


pearson_raw_quantile <- c(sort((pearson_residuals))[qqL],
                          sort((pearson_residuals))[qqU])
deviance_raw_quantile <- c(sort((deviance_residuals))[qqL],
                           sort((deviance_residuals))[qqU])
anscombe_raw_quantile <- c(sort((anscombe_residuals))[qqL],
                           sort((anscombe_residuals))[qqU])
regular_raw_quantile <- c(sort((regular_residuals))[qqL],
                          sort((regular_residuals))[qqU])



## PREDICTIONS ON HOLDOUT SET
dholdout_numeric <- model.matrix(~ . , data = dholdout)

preds <- predict(best_cv_model,
                 newx = as.matrix(dholdout_numeric[,-c(1,2)]),
                 s = best_lambda,
                 type = "response") 

predsD3 <- preds
responseD3 <- dholdout$CLM_AMT5


pearson_residuals_ho <- (
  dholdout$CLM_AMT5 - preds
) / (preds^(tweedie_power/2))
deviance_residuals_ho <- sqrt(
  tweedie.dev(dholdout$CLM_AMT5, preds, tweedie_power)
) * sign(dholdout$CLM_AMT5 - preds)
anscombe_residuals_ho <- (
  dholdout$CLM_AMT5^(1 - tweedie_power/3) - preds^(1 - tweedie_power/3)
) / (preds^(tweedie_power/6))
regular_residuals_ho <- dholdout$CLM_AMT5 - preds



## COVERAGE RATE
pearson_abs_rates <- mean(
  abs(pearson_residuals_ho) <= pearson_abs_quantile
)
deviance_abs_rates <- mean(
  abs(deviance_residuals_ho) <= deviance_abs_quantile
)
anscombe_abs_rates <- mean(
  abs(anscombe_residuals_ho) <= anscombe_abs_quantile
)
regular_abs_rates <- mean(
  abs(regular_residuals_ho) <= regular_abs_quantile
)

pearson_raw_rates <- mean(
  (pearson_residuals_ho <= pearson_raw_quantile[2]) &
    (pearson_residuals_ho >= pearson_raw_quantile[1])
)
deviance_raw_rates <- mean(
  (deviance_residuals_ho <= deviance_raw_quantile[2]) &
    (deviance_residuals_ho >= deviance_raw_quantile[1])
)
anscombe_raw_rates <- mean(
  (anscombe_residuals_ho <= anscombe_raw_quantile[2]) &
    (anscombe_residuals_ho >= anscombe_raw_quantile[1])
)
regular_raw_rates <- mean(
  (regular_residuals_ho <= regular_raw_quantile[2]) &
    (regular_residuals_ho >= regular_raw_quantile[1])
)

## LENGTH OF PREDICTION INTERVAL

pearson_abs_widths <-   mean(
  (preds + preds^(tweedie_power/2) * pearson_abs_quantile) -
    pmax(0, preds - preds^(tweedie_power/2) * pearson_abs_quantile)
)
regular_abs_widths <- mean((preds+regular_abs_quantile)
                           -pmax(0,preds-regular_abs_quantile))


pearson_raw_widths <- mean(
  pmax(0,preds + preds^(tweedie_power/2) * pearson_raw_quantile[2]) -
    pmax(0, preds + preds^(tweedie_power/2) * pearson_raw_quantile[1])
)
regular_raw_widths <- mean(pmax(0,preds+regular_raw_quantile[2])
                           -pmax(0,preds+regular_raw_quantile[1]))


anscombe_abs_widths <- mean(
  (
    preds^(1 - tweedie_power/3) + 
      anscombe_abs_quantile * preds^(tweedie_power/6)
  )^(3/(3 - tweedie_power)) - (
    pmax(0, preds^(1 - tweedie_power/3) - 
           anscombe_abs_quantile * preds^(tweedie_power/6))
  )^(3/(3 - tweedie_power))
)

anscombe_raw_widths <- mean(
  (
    pmax(0,preds^(1 - tweedie_power/3) + 
           anscombe_raw_quantile[2] * preds^(tweedie_power/6))
  )^(3/(3 - tweedie_power)) - (
    pmax(0, preds^(1 - tweedie_power/3) + 
           anscombe_raw_quantile[1] * preds^(tweedie_power/6))
  )^(3/(3 - tweedie_power))
)  

cat(sprintf("\nConformal Best Alpha: %.1f\n", best_alpha))
cat(sprintf("Conformal Best Tweedie Power: %.1f\n", best_tweedie_power))
cat(sprintf("Conformal Best Lambda: %f\n", best_lambda))



loglik <- numeric(length(tweedie_power_seq))
phi_vec <- numeric(length(tweedie_power_seq))
models <- vector("list", length(tweedie_power_seq))

for (k in seq_along(tweedie_power_seq)) {
  pp <- tweedie_power_seq[k]
  
  # Fit GLM with current Tweedie power
  temp_model <- glm(
    CLM_AMT5 ~ .,
    data = dtrain.test,
    family = tweedie(var.power = pp, link.power = 0)
  )
  
  # Predict means
  preds <- predict(temp_model, type = "response")
  
  # Extract dispersion parameter
  phi <- summary(temp_model)$dispersion
  
  # Store model and phi
  models[[k]] <- temp_model
  phi_vec[k] <- phi
  
  # Compute average log-likelihood
  loglik[k] <- mean(log(dtweedie(
    y = dtrain.test$CLM_AMT5,
    mu = preds,
    phi = phi,
    power = pp
  )))
}



best_index_simPI <- which.max(loglik)
best_power_simPI <- tweedie_power_seq[best_index_simPI]
best_model_simPI <- models[[best_index_simPI]]

cat(sprintf("SimPI Best Tweedie power: %.2f\n", best_power_simPI))

preds <- predict(
  object = best_model_simPI,
  newdata = dholdout,
  type = "response"
)

tt <- replicate(1000L,rtweedie(n=length(preds), power = best_power_simPI,
                               phi = phi_vec[best_index_simPI],
                               mu=preds))

simPI <-t(apply(tt,1,function(x) quantile(x,probs=c(0.025,0.975))))

simPI_widths <- mean(simPI[,2]-simPI[,1])

simPI_rates <- mean((simPI[,2]>=dholdout$CLM_AMT5) 
                    & (dholdout$CLM_AMT5>=simPI[,1])) 


intervalsD3_pearson_abs <- cbind(
  (preds + preds^(tweedie_power/2) * pearson_abs_quantile) ,
  pmax(0, preds - preds^(tweedie_power/2) * pearson_abs_quantile)
)[,c(2,1)]
intervalsD3_deviance_abs <-  cbind(
  (
    preds^(1 - tweedie_power/3) + 
      anscombe_abs_quantile * preds^(tweedie_power/6)
  )^(3/(3 - tweedie_power)) , (
    pmax(0, preds^(1 - tweedie_power/3) - 
           anscombe_abs_quantile * preds^(tweedie_power/6))
  )^(3/(3 - tweedie_power))
)[,c(2,1)]
intervalsD3_anscombe_abs <- cbind(
  (
    preds^(1 - tweedie_power/3) + 
      anscombe_abs_quantile * preds^(tweedie_power/6)
  )^(3/(3 - tweedie_power)) , (
    pmax(0, preds^(1 - tweedie_power/3) - 
           anscombe_abs_quantile * preds^(tweedie_power/6))
  )^(3/(3 - tweedie_power))
)[,c(2,1)]
intervalsD3_regular_abs <- cbind((preds+regular_abs_quantile)
                                 ,pmax(0,preds-regular_abs_quantile))[,c(2,1)]


intervalsD3_pearson_raw <- cbind(
  pmax(0,preds + preds^(tweedie_power/2) * pearson_raw_quantile[2]) ,
  pmax(0, preds + preds^(tweedie_power/2) * pearson_raw_quantile[1])
)[,c(2,1)]
intervalsD3_deviance_raw <- cbind(
  (
    pmax(0,preds^(1 - tweedie_power/3) + 
           anscombe_raw_quantile[2] * preds^(tweedie_power/6))
  )^(3/(3 - tweedie_power)) , (
    pmax(0, preds^(1 - tweedie_power/3) + 
           anscombe_raw_quantile[1] * preds^(tweedie_power/6))
  )^(3/(3 - tweedie_power))
)[,c(2,1)]
intervalsD3_anscombe_raw <- cbind(
  (
    pmax(0,preds^(1 - tweedie_power/3) + 
           anscombe_raw_quantile[2] * preds^(tweedie_power/6))
  )^(3/(3 - tweedie_power)) , (
    pmax(0, preds^(1 - tweedie_power/3) + 
           anscombe_raw_quantile[1] * preds^(tweedie_power/6))
  )^(3/(3 - tweedie_power))
)[,c(2,1)]
intervalsD3_regular_raw <- cbind(pmax(0,preds+regular_raw_quantile[2])
                                 ,pmax(0,preds+regular_raw_quantile[1]))[,c(2,1)]


# List of all interval matrices ----
interval_list <- list(
  pearson_raw = intervalsD3_pearson_raw,
  deviance_raw= intervalsD3_deviance_raw,
  anscombe_raw = intervalsD3_anscombe_raw,
  regular_raw = intervalsD3_regular_raw
)

# Check dimensions
sapply(interval_list, dim)

# Check heads (first 6 rows of each)
lapply(interval_list, head)

## saving glmnet ## -----
save(
  pearson_abs_rates, deviance_abs_rates, anscombe_abs_rates,
  pearson_abs_widths, deviance_abs_widths, anscombe_abs_widths,
  pearson_raw_rates, deviance_raw_rates, anscombe_raw_rates,
  pearson_raw_widths, deviance_raw_widths, anscombe_raw_widths,
  simPI_rates, simPI_widths,
  best_cv_model,
  best_alpha,
  best_lambda,
  best_tweedie_power,
  shuffle_index,
  best_index_simPI,
  best_power_simPI,
  best_model_simPI,
  regular_abs_rates,
  regular_raw_rates,
  regular_abs_widths,
  regular_raw_widths,
  
  regular_residuals_ho,
  pearson_residuals_ho,
  anscombe_residuals_ho,
  deviance_residuals_ho,
  
  
  responseD3,
  predsD3,
  
  
  intervalsD3_pearson_abs,
  intervalsD3_deviance_abs,
  intervalsD3_anscombe_abs,
  intervalsD3_regular_abs,
  
  intervalsD3_pearson_raw,
  intervalsD3_deviance_raw,
  intervalsD3_anscombe_raw,
  intervalsD3_regular_raw,
  
  
  pearson_abs_quantile,
  deviance_abs_quantile,
  anscombe_abs_quantile,
  regular_abs_quantile,
  
  pearson_raw_quantile,
  deviance_raw_quantile,
  anscombe_raw_quantile,
  regular_raw_quantile,
  
  pearson_residuals,
  deviance_residuals,
  anscombe_residuals,
  regular_residuals,
  
  file = paste0("sim_round_glmnet", j, ".rda")
)

##— Clean up —##
stopCluster(cl)
cat("Finished replicate", j, "\n")

