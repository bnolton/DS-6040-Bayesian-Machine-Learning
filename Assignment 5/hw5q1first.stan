data {
  int<lower=0> n;         
  array[n] int<lower=0> y;
}

parameters {
   real<lower=0> theta; // average parameter
}

model {
  // Prior
  theta ~ lognormal(0, 1); // lognomral with mu = 0 and sigma = 1
  // Likelihood
  y ~ poisson(theta); // 
}