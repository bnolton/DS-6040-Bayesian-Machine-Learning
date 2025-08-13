data {
  int<lower=0> N;                        // Number of observations
  int<lower=0> K;                        // Number of predictors
  matrix[N, K] X;                        // Predictor matrix 
  array[N] int<lower=0> y;               // Observed counts
  vector[N] off;                         // Offset on log scale 
}

parameters {
  real alpha;                            // Intercept
  vector[K] beta;                        // Coefficients
}

model {
  vector[N] eta;

  // Priors
  alpha ~ normal(0, 999);
  beta ~ normal(0, 10);

  // Manual linear predictor with offset
  eta = alpha + X * beta + off;

  // Likelihood with offset
  y ~ poisson_log(eta);
}

generated quantities {
  array[N] int y_tilde;
  for (n in 1:N) {
    real eta_n = alpha + dot_product(row(X, n), beta) + off[n];
    y_tilde[n] = poisson_log_rng(eta_n);
  }
}
