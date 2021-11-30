% CS534 - HW5
% Created by: Daniela Chanci Arrubla
% Date: 11/26/2021

% Problem 2 - Gap Statistics
close all; clear; clc;

% Load dataset
load ('HW5.mat');

% Define B and K
B = 100;
K = [1:20];

% Compute optimal k and gaps
[opt_k, gaps, desv] = gap_statistic(X, K, B);

d = zeros(1,20);
for i=1:length(d)
    d(1,i) = desv(10);
end

K_cond = [];

for i = 1:length(K)-1
    if gaps(i) >= gaps(i+1) - std(gaps(i))*sqrt(1 + 1/B)
        K_cond = [K_cond, i];
    end
end

% Plot error bar
figure;
errorbar(K, gaps, d)
xlabel('Clusters')
ylabel('Gap Statistic')
hold on

scatter(K_cond, gaps(i), 'bo')
scatter(opt_k,gaps(opt_k),'ro')