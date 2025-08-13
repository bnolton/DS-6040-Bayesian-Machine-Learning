data {
  int<lower=0> T;           
  array[T] real y;                 // time series data
}

parameters {
  real<lower=0> mu1;
  real<upper=0> mu2;
  array[2] real<lower=0> sigma;    // standard deviations of y
  array[2] simplex[2] Gamma; // state transition probabilities
  simplex[2] fz1;            // initial state probabilities
}

transformed parameters {

  array[2] real mu;
  mu[1] = mu1;
  mu[2] = mu2;


  // Forward algorithm log p(zt, y_{1:t})
  array[T] vector[2] logalpha;
  { 
    array[2] real accumulator;

    logalpha[1] = log(fz1) + normal_lpdf(y[1] | mu, sigma); // p(z1, y1)

    for (t in 2:T) {
      for (j in 1:2) { // j = current (t)
        for (i in 1:2) { // i = previous (t-1)
          accumulator[i] = logalpha[t-1, i] + log(Gamma[i, j]) + normal_lpdf(y[t] | mu[j], sigma[j]);
        }
        logalpha[t, j] = log_sum_exp(accumulator); // p(zt, y_{1:t})
      }
    }
  } // Forward
}

model {
  mu[1] ~ normal(0,10);
  mu[2] ~ normal(0,10);
  sigma ~ gamma(1,1);



  target += log_sum_exp(logalpha[T]); // Note: update based only on last logalpha
}