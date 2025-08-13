data {
  int<lower=0> T;   // # time points (equally spaced)
  vector[T] y;      // mean corrected return at time t
}
parameters {
  real mu;                     // mean log volatility
  real<lower=-1, upper=1> phi; // persistence of volatility
  real<lower=0> sigma;         // white noise shock scale
  vector[T] z;                 // log volatility at time t
}
model {
  phi ~ uniform(-1, 1);
  sigma ~ cauchy(0, 5);
  mu ~ cauchy(0, 10);
  z[1] ~ normal(mu, sigma / sqrt(1 - phi * phi));
  for (t in 2:T) {
    z[t] ~ normal(mu + phi * (z[t - 1] -  mu), sigma);
  }
  for (t in 1:T) {
    y[t] ~ normal(0, exp(z[t] / 2));
  }
}