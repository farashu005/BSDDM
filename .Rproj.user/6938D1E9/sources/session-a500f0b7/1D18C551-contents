init_Gen_b <- function(b, X) {
  set.seed(1561)
  noise <- rnorm(nrow(X), mean = 0, sd = 0.01)
  logb_vec <- log(b) + noise

  # Check for and handle NA or infinite values in X
  if (any(!is.finite(X))) {
    X[!is.finite(X)] <- 0  # Replace NA/Inf values with 0 or any other appropriate value
  }

  ## Performing ridge regression to estimate beta
  library(MASS)
  ridge_model <- lm.ridge(logb_vec ~ X - 1, lambda = seq(0, 1, by = 0.1), intercept = FALSE)  ## Without intercept
  # ridge_model <- lm.ridge(logb_vec ~ X , lambda = seq(0, 1, by = 0.1), intercept = FALSE)  ## With intercept
  optimal_lambda <- which.min(ridge_model$GCV)
  estimated_gama <- coef(ridge_model)[optimal_lambda, ]

  return(estimated_gama)
}

init_Gen_nu <- function(nu, X) {
  set.seed(1561)
  noise <- rnorm(nrow(X), mean = 0, sd = 0.01)
  lognu_vec <- log(nu) + noise
  # Check for and handle NA or infinite values in X
  if (any(!is.finite(X))) {
    X[!is.finite(X)] <- 0  # Replace NA/Inf values with 0 or any other appropriate value
  }
  ## Performing ridge regression to estimate beta
  # ridge_model <- lm.ridge(lognu_vec ~ X - 1, lambda = seq(0, 1, by = 0.1), intercept = FALSE)
  ridge_model <- lm.ridge(lognu_vec ~ X-1 , lambda = seq(0, 1, by = 0.1), intercept = FALSE)  ## Without intercept

  optimal_lambda <- which.min(ridge_model$GCV)
  estimated_beta <- coef(ridge_model)[optimal_lambda, ]

  return(estimated_beta)
}

