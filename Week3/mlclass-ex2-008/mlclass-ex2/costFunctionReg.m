function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X*theta);
part1 = -y .* log(h);
part2 = (1 - y) .* log(1 - h);
part3 = sum(theta(2:end) .^ 2);
J = sum(part1 - part2) / m + (part3 * lambda) / (2 * m);

%Compute regularized gradient
error = h - y;
grad_reg = (X' * error) / m + (lambda * theta(1:end)) / m;
grad_reg = grad_reg(2:end);
            

%Compute regular gradient
gradTheta0 = (X' * error) / m;
gradTheta0 = gradTheta0(1);
n = length(grad_reg);



%Compute final gradient
grad = [gradTheta0 ; zeros(n, 1)] + [0 ; grad_reg]


% =============================================================

end
