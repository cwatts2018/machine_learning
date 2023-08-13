function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

  m = size(X, 1);
  num_labels = size(Theta2, 1);
  
  X = [ones(m,1),X];
  p = zeros(size(X, 1), 1);
  z = X * (Theta1');
  a = sigmoid(z);
  a = [ones(m,1),a];
  z = a*(Theta2');
  [v,p] = max(sigmoid(z), [], 2);

end
