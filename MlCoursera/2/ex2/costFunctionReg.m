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

i= -y.*log(h)-(1-y).*log(1-h);

sumOfTheta2= sum(theta.^2)-theta(1)^2;

J = sum(i)/m+lambda/2/m*sumOfTheta2;




t =  X(:,1).*(h-y)/m;
g= sum(t);

q =  X(:,2:end).*(h-y)/m;
f= sum(q);
f=f'+ lambda/m*theta(2:end);
grad = [g ;f];







% =============================================================

end
