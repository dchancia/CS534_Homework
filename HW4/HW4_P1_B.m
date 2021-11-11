% HW4 - CS534 - Bagging and Boosting
% Created by: Daniela Chanci Arrubla
% Date: 11/06/2021

% Problem 1 - Part B

close all; clear; clc;

% Load data
load ionosphere;

% Convert Y to double
y = zeros(length(Y),1);
for i=1:length(Y)
    if strcmp(Y(i), 'g')
        y(i) = 1;
    else
        y(i) = 0;
    end 
end

% Adaboost from scratch

% Initialize weights
N = length(X);

% Define number of weak learners
weak_N = 300;

% Define nodes
lr = [1, 0.9];

for j=1:length(lr)
    % Initialize weights
    w = ones(N,1)*(1/N); 
    
    % Create accumulator
    robust_mdl = zeros(N,1);
    
    % Create accuracy vector
    acc = zeros(1, weak_N);
    
    % Adaboost.M1
    for i=1:weak_N
        tree = fitctree(X, y, 'MaxNumSplits', 5, 'Weights', w);
        CVtree = crossval(tree, 'KFold', 5);
        acc(1,i) = 1 - kfoldLoss(CVtree);
        y_hat = predict(tree, X);
        index = (y ~= y_hat);
        error = sum(w.*index)/sum(w);
        alpha = log((1 - error)/error);
        w = w.*exp(-alpha.*y.*y_hat);
        w = w./sum(w);
        robust_mdl = robust_mdl + alpha*y_hat;
    end
    
    % Plot accuracies
    plot([1:weak_N], acc, 'LineWidth', 1.5);
    ylim([0 1]);
    hold on
    % Final predictions
    y_hat = sign(robust_mdl);
    accuracy_final = sum(y_hat == y)/N;
    
end
hold off


lr = [1, 0.9];
for i=1:length(lr)
    t = templateTree('MaxNumSplits', 5);
    mdl = fitcensemble(X, y, 'Method', 'AdaBoostM1', 'Learners', t, ...
        'CrossVal', 'on', 'KFold', 5, 'NumLearningCycles', 300, 'LearnRate', lr(i));
    acc = 1 - kfoldLoss(mdl,'Mode','cumulative');
    plot(acc, 'LineWidth', 1.5);
    hold on
    ylim([0.7 1]);
end
legend({"Ordinary", "Shrunk"}, 'Location', 'southeast');
xlabel("Number of weak learners")
ylabel("Classification accuracy")
title("Adaboost M1")

