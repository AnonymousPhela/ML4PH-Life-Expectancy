
data {
    int<lower=1> N;                      // Number of observations
    int<lower=1> K;                      // Number of predictors
    int<lower=1> J;                      // Number of countries
    array[N] int<lower=1, upper=J> country; // Country index
    matrix[N, K] X;                      // Predictor matrix
    vector[N] y;                         // Life expectancy
}

parameters {
                             
    vector[K] beta;                      // Coefficients for health indicators
    vector[J] alpha;                    // Country-specific intercepts
    real<lower=0> sigma;                // Residual error
}

transformed parameters {
    vector[N] mu;
    for (i in 1:N) {
        mu[i] = alpha[country[i]] + X[i] * beta;
    }
}

model {
    // Priors

    beta ~ normal(0, 10);
    alpha ~ normal(0, 5);
    sigma ~ cauchy(0, 5);

    // Likelihood
    y ~ normal(mu, sigma);
}

generated quantities {
    vector[N] log_lik;
    vector[N] y_rep;
    for (i in 1:N) {
        log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
        y_rep[i] = normal_rng(mu[i], sigma);
    }
}
