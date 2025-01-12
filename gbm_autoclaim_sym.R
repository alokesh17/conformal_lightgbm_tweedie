rm(list = ls())
library(lightgbm)
library(tweedie)
# library(ggplot2)

filename <- "tweedie-autoclaim-sym"
##==== RANDOM SEED RECORDER ====##
if (!exists(".Random.seed")) runif(1)
initial_seed <- .Random.seed ## KEEP TRACK OF INITIAL SEED
save(initial_seed, file = paste0("RandomSeed-", filename, ".rda"))

## ##==== LOAD RECORDED RANDOM SEEDS ====##
## load(file = paste0("RandomSeed-", filename, ".rda"))
## assign(".Random.seed", initial_seed, .GlobalEnv)

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

best_model_evaluation <- function(dt) {
  ## USE CROSS-VALIDATION TO TUNE BEST TWEEDIE MODEL:
  ## 1. POWER PARAMETER TUNING WITH PROFILE LIKELIHOOD;
  ## 2. GIVEN EACH POWER PARAMETER, TRAIN `mu(x)` VIA LGB
  ##    AND ESTIMATE `phi` VIA MLE
  
  ## CREATE lgb.Dataset
  dtrain <- lgb.Dataset(
    data = as.matrix(dt$data),
    label = dt$label,
    categorical_feature = dt$categorical_feature
  )
  
  ## SEQUENCE OF CANDIDATE POWER PARAMETERS
  pow <- seq(1.2, 1.5, by = 0.025)
  
  ## AT EACH POWER PARAM: TRAIN `mu(x)` VIA LGB & ESTIMATE `phi` VIA MLE
  loglik <- rep(NA, length(pow))
  # model <- vector("list", length(pow))
  disp <- rep(NA, length(pow))
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
    
    ## REFIT MODEL WITH FULL DATA USING BEST NROUNDS
    fit <- lgb.train(
      params = params,
      data = dtrain,
      nrounds = best_iter
    )
    # model[[k]] <- fit
    preds <- predict(fit, as.matrix(dt$data))
    
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
      "Power = %.3f, best #iterations = %4d, dispersion = %.4f",
      pp, best_iter, phi
    ))
    
    ## CALCULATE TWEEDIE LOG-LIKELIHOOD
    loglik[k] <- mean(log(
      dtweedie(y = dt$label, mu = preds, phi = phi, power = pp)
    ))
    disp[k] <- phi
  }
  
  kmax <- which.max(loglik)
  print(
    sprintf(
      "Best power parameter = %.3f and dispersion = %.4f",
      pow[kmax], disp[kmax]
    )
  )
  
  # return(list(model = model[[kmax]], dispersion = disp[kmax]))
  return(list(power = pow[[kmax]], dispersion = disp[kmax]))
}

## LOAD AUTO CLAIM DATA
data(AutoClaim, package = "cplm")

## REMOVE FEATURES
x <- AutoClaim[, -c(1:5, 29)]

## INTEGER CODING TABLE FOR CATEGORICAL FEATURES
# coding <- sapply(x, function(x) {
#   if(is.factor(x))
#     data.frame(
#       code = seq_along(levels(x)),
#       level = levels(x)
#     )
# })
# coding[sapply(coding, is.null)] <- NULL
# print(coding)

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
qq <- ceiling((1 - alpha) * (n2 + 1)) / n2

nsim <- 100L
maxit <- 1500L
stepsize <- 25L
steps <- seq(stepsize, maxit, by = stepsize)
nsteps <- length(steps)
raw_rates       <- matrix(NA, nsteps, nsim)
pearson_rates   <- matrix(NA, nsteps, nsim)
deviance_rates  <- matrix(NA, nsteps, nsim)
anscombe_rates  <- matrix(NA, nsteps, nsim)
raw_widths      <- matrix(NA, nsteps, nsim)
pearson_widths  <- matrix(NA, nsteps, nsim)
deviance_widths <- matrix(NA, nsteps, nsim)
anscombe_widths <- matrix(NA, nsteps, nsim)

## LIGHTGBM PARAMETERS
nrounds <- 2000L
num_leaves <- 10L
learning_rate <- 0.005
nfold <- 5L

for (j in 1L:nsim) {
  cat(sprintf("\n==== Randomly splitting data... Round %3d... ====\n", j))
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
  
  ## TRAIN MODEL ON PROPER TRAINING SET
  fit <- best_model_evaluation(dtrain)
  # model <- fit$model
  # phi <- fit$dispersion
  # pp <- model$params$tweedie_variance_power
  pp <- fit$power
  
  ## REFIT MODEL WITH VARYING NUMBER OF TREES
  params <- list(
    num_leaves = num_leaves,
    learning_rate = learning_rate,
    objective = "tweedie",
    tweedie_variance_power = pp,
    verbose = 0L,
    use_missing = TRUE
  )
  
  model <- lgb.train(
    params = params,
    data = lgb.Dataset(
      data = as.matrix(dtrain$data),
      label = dtrain$label,
      categorical_feature = categorical_feature
    ),
    nrounds = maxit
  )
  
  for (i in seq(nsteps)) {
    ## CALCULATE CONFORMITY SCORES ON CALIBRATION SET
    preds <- predict(
      object = model,
      newdata = as.matrix(dtest$data),
      num_iteration = steps[i]
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
    
    ## CALCULATE THRESHOLDS OF CALIBRATION SET RESIDUALS
    raw_quantile <- quantile(abs(raw_residuals), qq)
    pearson_quantile <- quantile(abs(pearson_residuals), qq)
    deviance_quantile <- quantile(abs(deviance_residuals), qq, na.rm = TRUE)
    anscombe_quantile <- quantile(abs(anscombe_residuals), qq)
    
    ## PREDICTIONS ON HOLDOUT SET
    preds <- predict(
      object = model,
      newdata = as.matrix(dholdout$data),
      num_iteration = steps[i]
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
    
    ## COVERAGE RATE
    raw_rates[i, j] <- mean(
      abs(raw_residuals_ho) <= raw_quantile
    )
    pearson_rates[i, j] <- mean(
      abs(pearson_residuals_ho) <= pearson_quantile
    )
    deviance_rates[i, j] <- mean(
      abs(deviance_residuals_ho) <= deviance_quantile
    )
    anscombe_rates[i, j] <- mean(
      abs(anscombe_residuals_ho) <= anscombe_quantile
    )
    
    ## LENGTH OF PREDICTION INTERVAL
    raw_widths[i, j] <- mean(
      (preds + raw_quantile) -
        pmax(0, preds - raw_quantile)
    )
    pearson_widths[i, j] <- mean(
      (preds + preds^(pp/2) * pearson_quantile) -
        pmax(0, preds - preds^(pp/2) * pearson_quantile)
    )
    deviance_widths[i, j] <- mean(
      sapply(preds, root_search, thresh = deviance_quantile, p = pp)
    )
    anscombe_widths[i, j] <- mean(
      (
        preds^(1 - pp/3) + anscombe_quantile * preds^(pp/6)
      )^(3/(3 - pp)) - (
        pmax(0, preds^(1 - pp/3) - anscombe_quantile * preds^(pp/6))
      )^(3/(3 - pp))
    )
  }
}

save(
  raw_rates,
  pearson_rates,
  deviance_rates,
  anscombe_rates,
  raw_widths,
  pearson_widths,
  deviance_widths,
  anscombe_widths,
  file = paste0("Result-", filename, ".rda")
)
