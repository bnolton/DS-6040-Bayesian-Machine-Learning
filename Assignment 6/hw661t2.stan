data {
  int<lower=1> K;               // Dimension of multivariate normal
  int<lower=1> N;               // Number of data points

  real<lower=0> eta;            // LKJ concentration parameter (eta = 1 ⇒ uniform)
}

parameters {
  vector[K] mu;                            // Mean vector
  vector<lower=0>[K] L_std;                // Standard deviations
  cholesky_factor_corr[K] L_Omega;         // Cholesky factor of correlation matrix
}

transformed parameters {
  matrix[K, K] L_Sigma;

  // Construct Cholesky of covariance matrix: L_Sigma = diag(L_std) * L_Omega
  L_Sigma = diag_pre_multiply(L_std, L_Omega);
}

model {
  // Priors
  mu ~ normal(0, 5);
  L_std ~ normal(0, 2.5);
  L_Omega ~ lkj_corr_cholesky(eta);
}

generated quantities {
  matrix[K, N] y_sim;
    for (n in 1:N)
      y_sim[, n] = multi_normal_cholesky_rng(mu, L_Sigma);
}