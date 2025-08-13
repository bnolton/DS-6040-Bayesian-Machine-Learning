data {
  int<lower=1> T;        // Number of time points
  vector[T] y;           // Observed time series
}

parameters {
  real mu0;                // Initial state's mean
  real log_sigma_eps; // log Standard deviation of observation noise
  real log_sigma_eta; // log Standard deviation of state noise
}

transformed parameters {
  real<lower=0> sigma_eps = exp(log_sigma_eps); 
  real<lower=0> sigma_eta = exp(log_sigma_eta); 
}

model {
  // Priors
  mu0 ~ normal(0, 1); 
  log_sigma_eps ~ normal(0, 100);
  log_sigma_eta ~ normal(0, 100);

  // Kalman filter variables
  vector[T] mu_pred;   // Predicted state means
  vector[T] mu_filt;   // Filtered state means
  vector[T] P_pred;    // Predicted state variances
  vector[T] P_filt;    // Filtered state variances
  
  // Initial state
  mu_filt[1] = mu0;
  P_filt[1] = sigma_eta^2;
  mu_pred[1] = mu0;
  P_pred[1] = sigma_eta^2;
  
  // Kalman filter recursion
  for (t in 2:T) {
    // Prediction step
    mu_pred[t] = mu_filt[t-1];
    P_pred[t] = P_filt[t-1] + sigma_eta^2;
    
    // Update step
    real K = P_pred[t] / (P_pred[t] + sigma_eps^2); // Kalman gain
    mu_filt[t] = mu_pred[t] + K * (y[t] - mu_pred[t]);
    P_filt[t] = (1 - K) * P_pred[t];
  }

  // Likelihood
  for (t in 1:T) {
    target += normal_lpdf(y[t] | mu_pred[t], sqrt(P_pred[t] + sigma_eps^2));
  }
}
