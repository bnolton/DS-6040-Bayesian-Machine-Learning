data {
  int<lower=1> N;                // number of data points
  array[N] int<lower=0> y;       // successes
  array[N] int<lower=0> n;       // trials
  //real mu_prior;                 // prior mean for logit(theta)
  //real<lower=0> sigma_prior;     // prior std for logit(theta)
}

parameters {
  real theta_raw;                // unconstrained parameter (logit scale)
}

transformed parameters {
  real<lower=0, upper=1> theta;  // probability of success
  theta = inv_logit(theta_raw);
}

model {
  // Prior
  theta_raw ~ normal(0, 1);

  // Likelihood
    y ~ binomial(n, theta);
}
