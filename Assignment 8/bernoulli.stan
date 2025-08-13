data {
  int<lower=0> N;          
  int<lower=0> K;           
  matrix[N, K] X;           
  int<lower=0, upper=1> y[N];  
}

parameters {
  real beta_intercept;    // intercept
  vector[K] predictor_betas;          
}

model {
  // Priors
  beta_intercept ~ normal(0, 999);
  predictor_betas ~ normal(0, 10);
  
  // Logistic regression likelihood
  y ~ bernoulli_logit(beta_intercept + X * predictor_betas);
}