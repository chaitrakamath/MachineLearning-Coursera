function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X * theta;
part1 = sum((h - y) .^2);
part2 = sum(theta(2:end) .^ 2);
J = (part1 + lambda * part2) / (2 * m);

%Compute regularized gradient
error = h - y;
grad_reg = (X' * error) / m + (lambda * theta(1:end)) / m;
grad_reg = grad_reg(2:end);
            
            
%Compute regular gradient
gradTheta0 = (X' * error) / m;
gradTheta0 = gradTheta0(1);
n = length(grad_reg);
                          
                          
                          
%Compute final gradient
grad = [gradTheta0 ; zeros(n, 1)] + [0 ; grad_reg];







% =========================================================================

grad = grad(:);

end
