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

## CREATE LIGHTGBM DATASET
dt <- list(data = as.matrix(x), label = AutoClaim$CLM_AMT5 / 1000)
dtrain <- lgb.Dataset(
  data = dt$data,
  label = dt$label,
  categorical_feature = colnames(x)[which(index)]
)

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
