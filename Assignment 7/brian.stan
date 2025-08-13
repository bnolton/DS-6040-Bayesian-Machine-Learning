data {
  int<lower=0> N;         // number of data points
  int<lower=1> D;         // dimensionality of each data point
  matrix[N, D] y;         // data matrix (each row is a data point)
}

parameters {
  simplex[3] lambda;        // mixture weights
  array[3] vector[D] mu;    // a mean vector for each cluster
  array[3] cholesky_factor_corr[D] L;             // Cholesky factors for each component
  array[3] vector<lower=0>[D] sigma;              // scale vectors for each component
}

transformed parameters {
  array[3] matrix[D, D] Sigma_chol;               // Cholesky-decomposed covariances
  for (k in 1:3) {
    Sigma_chol[k] = diag_pre_multiply(sigma[k], L[k]);
  }
}

model {
  vector[3] lps;

  // Priors
  for (k in 1:3) {
    mu[k] ~ normal(0, 2);
    sigma[k] ~ cauchy(0, 2.5);
    L[k] ~ lkj_corr_cholesky(2.0);
  }

  // Marginalized mixture likelihood
  for (n in 1:N) {
    for (k in 1:3) {
      lps[k] = log(lambda[k]) + 
               multi_normal_cholesky_lpdf(y[n] | mu[k]', Sigma_chol[k]);
    }
    target += log_sum_exp(lps);
  }
}

generated quantities {
  array[N] int<lower=1, upper=3> z;   // latent assignments
  matrix[N, D] y_rep;                 // posterior predictive samples


  for (n in 1:N) {
    z[n] = categorical_rng(lambda);
    y_rep[n] = (multi_normal_cholesky_rng(mu[z[n]]', Sigma_chol[z[n]]))';
  }
}