data {
  int<lower=0> n;         
  array[n] int<lower=0> y;
}

parameters {
   real<lower=0> theta; // average parameter
}

transformed parameters {
    real<lower=0> exp_theta;
    exp_theta = exp(theta);
}

model {
  // Prior
  theta ~ normal(120, 5); // normal with mu = 0 and sigma = 1
  // Likelihood
  y ~ poisson(exp_theta); // 
}