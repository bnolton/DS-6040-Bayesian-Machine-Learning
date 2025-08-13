data {
  int<lower=0> N;         // number of data points
  int<lower=1> D;         // dimensionality
  matrix[N, D] y;         // data
}
 
parameters {
  simplex[3] lambda;            // mixture weights
  ordered[3] mu_dim1;           // ordered means on dim1
  matrix[3, D-1] mu_rest;       // remaining coordinates
  array[3] cov_matrix[D] Sigma; // cluster-specific covariances
}
 
transformed parameters {
  array[3] vector[D] mu;
  for (k in 1:3) {
    mu[k,1] = mu_dim1[k];
    for (d in 2:D)
      mu[k,d] = mu_rest[k,d-1];
  }
}
 
model {
  array[3] real ps;
  vector[3] alpha = rep_vector(5.0, 3);

  // Priors
  lambda ~ dirichlet(alpha);
  for (k in 1:3) {
    mu[k] ~ normal(0, 2);
    Sigma[k] ~ inv_wishart(D + 1, diag_matrix(rep_vector(1, D)));
  }
 
  // soft constraint to encourage separation along dim1
  for (k in 2:3) {
    target += normal_lpdf(mu_dim1[k] - mu_dim1[k-1] | 5.0, 0.25);
  }
 
  // Likelihood
  for (n in 1:N) {
    for (k in 1:3) {
      ps[k] = log(lambda[k]) + multi_normal_lpdf(y[n] | mu[k], Sigma[k]);
    }

    target += log_sum_exp(ps);
  }
}
 
generated quantities {
  matrix[N, 3] label_prob;

  for (n in 1:N) {
    array[3] real log_ps;
    real max_log_ps;
    for (k in 1:3)
      log_ps[k] = log(lambda[k]) + multi_normal_lpdf(y[n] | mu[k], Sigma[k]);
    max_log_ps = max(log_ps);
    for (k in 1:3)
      label_prob[n, k] = exp(log_ps[k] - max_log_ps);
    label_prob[n] /= sum(label_prob[n]);
  }
}