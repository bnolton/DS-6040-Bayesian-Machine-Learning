data {
  int<lower=0> J;               // number of rockets
  array[J] int<lower=0> n;      // number of launches for each rocket
  array[J] int<lower=0> y;      // number of failures for each rocket
}

parameters {
  real<lower=0> alpha;                // alpha parameter for the Beta distribution
  real<lower=0> beta;                 // beta parameter for the Beta distribution
  array[J] real<lower=0, upper=1> z;  // failure probabilities for each group
}

model {
  // Priors
  alpha ~ gamma(1, 1);    // Gamma prior for alpha
  beta ~ gamma(1, 1);     // Gamma prior for beta
  
  // Complete-Data Likelihood
  for (j in 1:J) {
    z[j] ~ beta(alpha, beta);     // hierarchical model
    y[j] ~ binomial(n[j], z[j]);  // binomial likelihood
  }
}

generated quantities {
  array[J] real y_pred_rate;  // posterior predictive counts

  for (j in 1:J) {
    int y_sim = binomial_rng(n[j], z[j]);  // simulate failures
    y_pred_rate[j] = y_sim * 1.0 / n[j];     //convert to failure rate
  }
}
