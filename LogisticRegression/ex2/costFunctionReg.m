function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

%theta0 = [0; theta(2:end)];
%
%h = sigmoid(X * theta);
%J = (1/m) * sum(-y .* log(h) - (1 - y) .* log(1 - h)) + ...
%    lambda / (2*m) * theta0' * theta0;
%grad = (1/m) * (X' * (h - y) + lambda * theta0);

hypothesis = sigmoid(X * theta);
theta1 = [0 ; theta(2:end)];

% lambda / 2*m != lambda / (2*m)
regularized = (lambda / (2*m)) * (theta1' * theta1);

J = (1/m) * sum(-y .* log(hypothesis) - (1-y) .* log(1-hypothesis)) + regularized;
grad = (1/m) * (X' * (hypothesis - y) + lambda * theta1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
