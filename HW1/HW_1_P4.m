% Daniela Chanci Arrubla
% HW1 CS534 - Problem 4

clear;clc;
load('HW1_4');

%% Problem 4a

% Plot samples and given f(x)
figure;
plot(X, Y, 'k.');
hold on
fplot(@(x) alpha(1)*x^3 + alpha(2)*x^2 + alpha(3)*x^2 + alpha(4), [-1 1],...
    'b', 'LineWidth', 2);

% Polynomial least squares
M = zeros(length(X), length(alpha));  % Build matrix
M(:,1) = 1;
for i=1:3
    M(:,i+1) = X.^i
end
alpha_hat = inv(M'*M)*M'*Y;  % Solve for alpha

%Plot f_hat
fplot(@(x) alpha_hat(1)*x^3 + alpha_hat(2)*x^2 + alpha_hat(3)*x^2 +...
    alpha_hat(4), [-1 1], 'r', 'LineWidth', 2);
hold off
legend('Samples', 'True Model', 'Estimated Model')

%% Problem 4b

points = [X, Y];
sampleSize = 5;
threshold = 0.3;
fitLineFcn = @(points) polyfit(points(:,1),points(:,2),1);
evalLineFcn = @(model, points) sum((points(:, 2) - polyval(model, points(:,1))).^2,2);
[modelRANSAC, inlierIdx] = ransac(points,fitLineFcn,evalLineFcn,sampleSize,threshold);
modelInliers = polyfit(points(inlierIdx,1),points(inlierIdx,2),1);
inlierPts = points(inlierIdx,:);
x = [min(inlierPts(:,1)) max(inlierPts(:,1))];
y = modelInliers(1)*x + modelInliers(2);

% Plot samples and given f(x)
figure;
plot(X, Y, 'k.');
hold on
fplot(@(x) alpha(1)*x^3 + alpha(2)*x^2 + alpha(3)*x^2 + alpha(4), [-1 1],...
    'b', 'LineWidth', 2);
plot(x, y, 'r', 'LineWidth', 2)
hold off
legend('Samples', 'True Model', 'Estimated Model')