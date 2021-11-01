% Daniela Chanci Arrubla
% HW1 CS534 - Problem 2

clear;clc;

%% Problem 2b

% Create samples from Standard Normal Distribution
mu = [0 0]';
Rw = [1 0; 0 1];
W = mvnrnd(mu,Rw,1000);
W = W';

% Create samples from given distribution
Rx = [1 -0.5; -0.5 0.5]; % Covariance
mu = [1 1];  % Mean
[V,D] = eig(Rx); % Eigenvalues and Diagonal Matrix
D1 = D^(1/2);
X = V*D1*W;  % New Samples
X = X + 1;  % New Samples

% Plot 1000 samples
figure; 
plot(X(1,:),X(2,:),'k.', 'MarkerSize', 10);
title('Problem 2b', 'FontSize', 14);
axis('equal');
hold on

% Plot eigenvectors
V = V*D + 1;  
plot([1 V(1,1)],[1 V(2,1)], 'r', 'LineWidth', 2);
plot([1 V(1,2)],[1 V(2,2)], 'r', 'LineWidth', 2);

% Plot level curves of PDF
G = gmdistribution(mu,Rx);
f = @(x,y) pdf(G,[x y]);
fcontour(f, 'LineWidth', 1);
hold off

%% Problem 2c

% Plot 1000 samples
figure; 
plot(X(1,:),X(2,:),'k.', 'MarkerSize', 10);
title('Problem 2c', 'FontSize', 14);
axis('equal');
hold on

% Plot level curves of euclidean distance
f = @(x,y) sqrt((x-mu(1))^2 + (y-mu(2))^2);
fcontour(f,'LineWidth',1);
hold off

%% Problem 2d

% Plot 1000 samples
figure; 
plot(X(1,:),X(2,:),'k.', 'MarkerSize', 10);
title('Problem 2d', 'FontSize', 14);
axis('equal');
hold on

% Plot level curves of Mahalanobis distance
f = @(x,y) sqrt(([x;y]-mu')'*inv(Rx)*([x;y]-mu'));
fcontour(f,'LineWidth',1);
hold off
