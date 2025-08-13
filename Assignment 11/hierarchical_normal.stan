data {
  int<lower=0> J;               // number of schools
  array[J] real y;              // estimated treatment effects
  array[J] real<lower=0> sigma; // standard error of effect estimates
}

parameters {
  real mu;               // parameter: shared mean of all treatment effects
  real<lower=0> tau;     // parameter: shared standard deviation of treatment effects
  vector[J] z;       // treatment effects
}

model {
  // prior
  mu ~ normal(0, 5); // A weakly informative prior for the grand mean
  tau ~ cauchy(0, 5); // A weakly informative prior for the standard deviation

  // Complete-Data Likelihood
  z ~ normal(mu, tau); 
  y ~ normal(z, sigma); 
}
