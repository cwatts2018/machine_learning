function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
  
  % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  % for our 2 layer neural network
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));
  
  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));
  
  % Setup some useful variables
  m = size(X, 1);
           
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

  %Foward propagation
  y_matrix = eye(num_labels)(y,:);
  
  a1 = [ones(m,1),X];
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  
  a2 = [ones(m,1), a2];
  z3 = a2 * Theta2';
  h = sigmoid(z3);
  
  %Unregularized Cost
  J = 1/m * (sum(sum(-y_matrix .*log(h))) - sum(sum((1-y_matrix).*log(1-h))));
  
  %Regularized Cost
  Theta1(:,1) = 0;
  Theta2(:,1) = 0;
  J = J + (lambda/(2*m) * (sum(sum(Theta1.*Theta1))+sum(sum(Theta2.*Theta2))));
  size(y_matrix);
  size(h);
  
  %Back propagation (Unregularized gradient)
  delta3 = h-y_matrix;
    
  delta2 = delta3*Theta2(:,2:end) .* sigmoidGradient(z2);
    
  t = delta2' * a1;
  t2 = delta3' * a2;
  
  Theta1_grad = 1/m * t;
  Theta2_grad = 1/m * t2;
  grad = [Theta1_grad(:); Theta2_grad(:)];

  %Regularized Gradient
  Theta1(:,1) = 0;
  Theta2(:,1) = 0;
  r1 = (lambda/m) * Theta1;
  r2 = (lambda/m) * Theta2;
  Theta1_grad = Theta1_grad + r1;
  Theta2_grad = Theta2_grad + r2;
  grad = [Theta1_grad(:); Theta2_grad(:)];

  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
