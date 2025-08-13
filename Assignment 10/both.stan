data {
  int<lower=1> N;               // number of observations
  int<lower=1> D;               // number of features
  matrix[N, D] y;               // observed data
}

parameters {
  simplex[3] lambda;                              // mixing proportions
  ordered[3] mu_1;                                // ordered means on dim1
  matrix[3, D-1] mu_rest;                         // remaining coordinates
  array[3] cholesky_factor_corr[D] L;             // Cholesky factors for each component
  array[3] vector<lower=0>[D] sigma;              // scale vectors for each component
}

transformed parameters {
  array[3] vector[D] mu;
  for (k in 1:3) {
    mu[k,1] = mu_1[k];
    for (d in 2:D)
      mu[k,d] = mu_rest[k,d-1];
  }

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
  matrix[N, D] label_prob;                 // posterior predictive samples


  for (n in 1:N) {
    z[n] = categorical_rng(lambda);
    label_prob[n] = (multi_normal_cholesky_rng(mu[z[n]]', Sigma_chol[z[n]]))';
  }
}

