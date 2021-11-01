% Daniela Chanci Arrubla
% HW2 CS534 - Problem 3

clear;clc;

%% Problem 3a

% Create training samples from given 
mu = [0 0]';
Rw = [1 0; 0 1];
W = mvnrnd(mu,Rw,10);
W = W';

% Create 10 samples from given distribution 1
Rx1 = [2 -1; -1 2]; % Covariance
mu1 = [2 2]';  % Mean
[V1,D1] = eig(Rx1); % Eigenvalues and Diagonal Matrix
D1 = D1^(1/2);
X1 = V1*D1*W;  % New Samples
X1 = X1 + 2;  % New Samples

% Create 10 samples from given distribution 2
Rx2 = [1 0.5; 0.5 1]; % Covariance
mu2 = [0 0]';  % Mean
[V2,D2] = eig(Rx2); % Eigenvalues and Diagonal Matrix
D2 = D2^(1/2);
X2 = V2*D2*W;  % New Samples

% Plot samples
figure; 
plot(X1(1,:),X1(2,:),'.', 'MarkerSize', 14, 'Color', [0.4660 0.6740 0.1880]);
title('Problem 3a - QDA', 'FontSize', 14);
axis('equal');
hold on

plot(X2(1,:),X2(2,:),'.', 'MarkerSize', 14, 'Color', [0.4940 0.1840 0.5560]);

% Decision Boundaries

% QDA
s_mu1 = mean(X1')';
s_mu2 = mean(X2')';
s_Rx1 = cov(X1');
s_Rx2 = cov(X2');
syms x1 x2
qda = -0.5*log(det(s_Rx1)) - 0.5*([x1; x2] - s_mu1)'*inv(s_Rx1)*([x1; x2] - s_mu1)...
    + log(0.5) == -0.5*log(det(s_Rx2)) - 0.5*([x1; x2] - s_mu2)'*inv(s_Rx2)*...
    ([x1; x2] - s_mu2) + log(0.5);
fimplicit(qda, '-', 'LineWidth', 1, 'Color', [0 0.4470 0.7410]) % fimplicit...
% instead of solve to discard asymtotes

% Theoretical Bayes
syms x1 x2
tbdb = (1/((det(Rx1))^(1/2)))*exp(-(1/2)*([x1 x2]' - mu1)'*inv(Rx1)*...
    ([x1 x2]' - mu1)) == (1/((det(Rx2))^(1/2)))*exp(-(1/2)*...
    ([x1 x2]' - mu2)'*inv(Rx2)*([x1 x2]' - mu2));
fimplicit(tbdb, '-k', 'LineWidth', 1)

% Empirical Bayes
ebdb = (1/((det(s_Rx1))^(1/2)))*exp(-(1/2)*([x1 x2]' - s_mu1)'*inv(s_Rx1)*...
    ([x1 x2]' - s_mu1)) == (1/((det(s_Rx2))^(1/2)))*exp(-(1/2)*...
    ([x1 x2]' - s_mu2)'*inv(s_Rx2)*([x1 x2]' - s_mu2));
fimplicit(ebdb, '--', 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980])
legend('X_1', 'X_2', 'QDA', 'Theoretical Bayes', 'Empirical Bayes');
hold off

%% Problem 3b

% Create training samples from given 
W = mvnrnd(mu,Rw,1000);
W = W';

% Create 10 samples from given distribution 1
X1 = V1*D1*W;  % New Samples
X1 = X1 + 2;  % New Samples

% Create 10 samples from given distribution 2
X2 = V2*D2*W;  % New Samples

% Plot samples
figure; 
plot(X1(1,:),X1(2,:),'.', 'MarkerSize', 6, 'Color', [0.4660 0.6740 0.1880]);
title('Problem 3b - QDA', 'FontSize', 14);
axis('equal');
hold on

plot(X2(1,:),X2(2,:),'.', 'MarkerSize', 6, 'Color', [0.4940 0.1840 0.5560]);

% Decision Boundaries

% QDA
s_mu1 = mean(X1')';
s_mu2 = mean(X2')';
s_Rx1 = cov(X1');
s_Rx2 = cov(X2');
syms x1 x2
qda = -0.5*log(det(s_Rx1)) - 0.5*([x1; x2] - s_mu1)'*inv(s_Rx1)*([x1; x2] - s_mu1)...
    + log(0.5) == -0.5*log(det(s_Rx2)) - 0.5*([x1; x2] - s_mu2)'*inv(s_Rx2)*...
    ([x1; x2] - s_mu2) + log(0.5);
fimplicit(qda, '-', 'LineWidth', 1, 'Color', [0 0.4470 0.7410]) % fimplicit...
% instead of solve to discard asymtotes

% Theoretical Bayes
syms x1 x2
tbdb = (1/((det(Rx1))^(1/2)))*exp(-(1/2)*([x1 x2]' - mu1)'*inv(Rx1)*...
    ([x1 x2]' - mu1)) == (1/((det(Rx2))^(1/2)))*exp(-(1/2)*...
    ([x1 x2]' - mu2)'*inv(Rx2)*([x1 x2]' - mu2));
fimplicit(tbdb, '-k', 'LineWidth', 1)

% Empirical Bayes
ebdb = (1/((det(s_Rx1))^(1/2)))*exp(-(1/2)*([x1 x2]' - s_mu1)'*inv(s_Rx1)*...
    ([x1 x2]' - s_mu1)) == (1/((det(s_Rx2))^(1/2)))*exp(-(1/2)*...
    ([x1 x2]' - s_mu2)'*inv(s_Rx2)*([x1 x2]' - s_mu2));
fimplicit(ebdb, '--', 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980])
legend('X_1', 'X_2', 'QDA', 'Theoretical Bayes', 'Empirical Bayes');
hold off