data {
  int<lower=0> n; // 
  array[n] int<lower=0> y; // 
}

parameters {
   real logit(theta); // average parameter
}

model {
  // Prior
  theta ~ normal(0, 1); // normal with mu = 0 and sigma = 1
  // Likelihood
  y ~ binomial(n, theta);
}