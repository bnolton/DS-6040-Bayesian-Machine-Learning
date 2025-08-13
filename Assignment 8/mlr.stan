data {
  int<lower=0> N;        // Number of observations/rows
  int<lower=0> K;        // Number of predictors (including intercept!)
  matrix[N, K] X;        // Predictor matrix (including intercept!)
  vector[N] y;           // Response variable
}

parameters {
  vector[K] beta;        
  real<lower=0> sigma;   // Standard deviation instead of variance
}

model {
  beta ~ normal(0, 10);
  sigma ~ lognormal(0, 100);
  
  for(i in 1:N){
    target += normal_lpdf(y[i] | X[i,] * beta, sigma);
    }
}

generated quantities {
    vector[N] y_tilde;
    
    for(i in 1:N){
        y_tilde[i] = normal_rng(X[i,] * beta, sigma);
    }
}