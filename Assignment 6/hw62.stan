data {
  int<lower=1> K;                 // Dimension of the multivariate normal
  int<lower=1> N;                 // Number of prior-predictive draws for y

  // --- NIW hyper-parameters ---
  vector[K] mu0;                  // Location for the prior mean
  real<lower=0> kappa0;           // Mean “sample size” (larger ⇒ tighter on mu)
  real<lower=K-1> nu0;             // Degrees of freedom for Inverse-Wishart
  matrix[K, K] Lambda0;               // Scale matrix for Inverse-Wishart
}

generated quantities {
  cov_matrix[K] Sigma;            // Drawn covariance matrix
  vector[K]    mu;                // Drawn mean vector
  matrix[N, K] y_sim;             // Prior-predictive samples

  // 1. Draw Σ ~ InvWishart(ν, Ψ)
  Sigma = inv_wishart_rng(nu0, Lambda0);

  // 2. Draw μ | Σ  ~  MVN( μ₀ , Σ / κ₀ )
  mu = multi_normal_rng(mu0, Sigma / kappa0);

  // 3. Draw N synthetic observations  y | μ, Σ  ~  MVN( μ , Σ )
  for (n in 1:N)
    y_sim[, n] = multi_normal_rng(mu, Sigma);
}
