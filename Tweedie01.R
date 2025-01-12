rm(list = ls())
library(lightgbm)
library(ggplot2)
library(tweedie)

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
#

data(AutoClaim, package = "cplm")
## FOLLOWING YIP AND YAU (2005)
auto <- AutoClaim[AutoClaim$IN_YY == TRUE, ]

##===== LIGHTGBM HANDLES MISSING VALUES BY DEFAULT =====##
# ## FILL IN MISSING VALUES
# auto$YOJ[which(is.na(auto$YOJ))] <- 0
# auto$HOME_VAL[which(is.na(auto$HOME_VAL))] <- 0
# auto$SAMEHOME[which(is.na(auto$SAMEHOME))] <- 1

## REMOVE FEATURES
x <- auto[, -c(1:5, 10, 16, 29)]

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

## ENCODE CATEGORICAL FEATURES
index <- sapply(x, is.factor)
x[index] <- lapply(x[index], as.integer)

dt <- list(data = as.matrix(x), label = auto$CLM_AMT5 / 1000)
dtrain <- lgb.Dataset(
  data = dt$data,
  label = dt$label,
  categorical_feature = colnames(x)[which(index)]
)

pow <- seq(1.2, 1.5, by = 0.01)
nrounds <- 1000L
num_leaves <- 10L
learning_rate <- 0.005
nfold <- 5L

loglik <- rep(NA, length(pow))
model <- vector("list", length(pow))
set.seed(12321)
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
