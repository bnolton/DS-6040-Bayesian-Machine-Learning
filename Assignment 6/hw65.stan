data {
  int<lower=1> n;             // Number of data points
  int<lower=2> d;             // Dimension of the data
  matrix[d, n] y;             // Data matrix
  vector<lower=0>[d] diag_cov;  // Diagonal elements of the prior covariance matrix
}

parameters {
  vector[d] mu;                // mean
  cov_matrix[d] Sigma;          // covariance matrix
}

model {
  Sigma ~ inv_wishart(10, diag_matrix(diag_cov)); // Prior on the covariance matrix
  mu ~ multi_normal(rep_vector(0, d), Sigma);         // Prior on the mean

  // Likelihood
  for (i in 1:n) {
    y[,i] ~ multi_normal(mu, Sigma); // Likelihood of the data
  }
}

generated quantities {
    vector[d] y_tilde;
    y_tilde = multi_normal_rng(mu, Sigma);
}