data {
  int<lower=0> N;                 // Number of observations
  int<lower=0> K;                 // Number of predictors
  matrix[N, K] X;                 // Design matrix
  array[N] int<lower=0> y;        // Count outcome variable
}

parameters {
  real alpha;                     // intercept
  vector[K] beta;                 // Regression coefficients
}

model {
  // Priors
  alpha ~ normal(0, 999);
  beta ~ normal(0, 10);

  // Likelihood
  y ~ poisson_log_glm(X, alpha, beta);
}

generated quantities {
  array[N] int y_tilde;
  for (n in 1:N) {
    real eta = alpha + dot_product(row(X, n), beta);    // log lambda
    y_tilde[n] = poisson_log_rng(eta);                  // simulate new observation
  }
}