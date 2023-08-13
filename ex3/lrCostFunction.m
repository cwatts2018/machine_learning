function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

  m = length(y); % number of training examples
  J = 0;
  grad = zeros(size(theta));
  
  %Unregularized
  %h = sigmoid(X * theta);
  %J = 1/m * -y' * log(h) - (1-y)' * log(1-h);
  %grad = 1/m * (h-y);
  
  %Regularized
  h = sigmoid(X * theta);
  theta(1) = 0;
  J = 1/m * (-y' * log(h) - (1-y)' * log(1-h)) + lambda/(2*m) * (theta'*theta);
  grad = 1/m * X' * (h-y) + lambda/m * theta;
  
  grad = grad(:);

end
