% Daniela Chanci Arrubla
% HW1 CS534 - Problem 3

clear;clc;
load('HW1_3');

%% Problem 3a
% Lasso using gradient descent

lambda = 1;  % penalty weight
lr = 5e-3;  % learning rate
epochs = 10000;  % gradiente updates
losses = zeros(epochs, 1);

rng('default');
s = rng;
[N, p] = size(X);
beta_lasso = randn(p, 1); % initialization beta
s_beta = zeros(p, 1);

% Train the LASSO regression
for i=1:epochs
    Y_hat = X*beta_lasso;
    loss = sum((Y_hat - Y).^2) + lambda*(norm(beta_lasso,1));
    losses(i) = loss;
    error = Y_hat - Y;
    % Gradient updates
    grad = lr*((1/N)*2*(X')*error + lambda*sign(beta_lasso));
    beta_lasso = beta_lasso - grad;
end

% Plot Final Coefficients
figure;
plot([1:length(beta_lasso)], beta_lasso, 'k.', 'MarkerSize', 12);
xlabel('Coefficients Index', 'FontSize', 16);
ylabel('Coefficients Magnitude', 'FontSize', 16);
title('Final Model Coefficientes Lasso', 'FontSize', 16)

%% Problem 3b
% Least squares model without regularization

lr = 5e-3;  % learning rate
epochs = 10000;  % gradiente updates
losses = zeros(epochs, 1);

[N, p] = size(X);
beta = randn(p, 1); % initialization beta

% Train the regression
for i=1:epochs
    Y_hat = X*beta;
    loss = sum((Y_hat - Y).^2);
    losses(i) = loss;
    error = Y_hat - Y;
    % Gradient updates
    grad = (lr/N)*(2*(X')*error);
    beta = beta - grad;
end

%Plot Final Coefficients
figure;
plot([1:length(beta)], beta, 'k.', 'MarkerSize', 12);
xlabel('Coefficients Index', 'FontSize', 16);
ylabel('Coefficients Magnitude', 'FontSize', 16);
title('Final Model Coefficientes', 'FontSize', 16)

%% Problem 3c
% Comparison of the two models

% Least Squares
mse1 = (1/N)*sum((X_test*beta - Y_test).^2)

% Lasso
mse2 = (1/N)*sum((X_test*beta_lasso - Y_test).^2)

% The mean squared error obtained with LASSO is less than the one 
% obtained with the Least Squares model