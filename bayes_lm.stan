
data {
    int<lower=1> N;                // number of observations
    int<lower=1> K;                // number of predictors
    matrix[N, K] X;                // predictor matrix
    vector[N] y;                   // life expectancy (outcome variable)
}

parameters {
    real beta0;                    // intercept
    vector[K] beta;                 // regression coefficients
    real<lower=0> sigma;           // residual error
}

transformed parameters {
    vector[N] mu;
    mu = beta0 + X * beta;         // linear model
}

model {
    beta0 ~ normal(60, 10);          // prior for intercept
    beta ~ normal(0, 10);           // prior for coefficients
    sigma ~ cauchy(0, 5);          // prior for residual standard deviation
    y ~ normal(mu, sigma);         // likelihood
}

generated quantities {
    vector[N] log_lik;             // log-likelihood for each data point
    array[N] real y_rep;           // posterior predictive values

    for (i in 1:N) {
        log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
        y_rep[i] = normal_rng(mu[i], sigma); // posterior predictive
    }
}
