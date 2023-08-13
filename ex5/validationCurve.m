function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

    % Selected values of lambda
    lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
    
    error_train = zeros(length(lambda_vec), 1);
    error_val = zeros(length(lambda_vec), 1);
    
    for i = 1:length(lambda_vec)
      theta = trainLinearReg(X,y,lambda_vec(i,:));
      [J, grad] = linearRegCostFunction(X, y, theta, 0);
      [J1, grad1] = linearRegCostFunction(Xval, yval, theta, 0);
      
      error_train(i,:) = J;
      error_val(i,:) = J1;
    end;
    
    %prints min error_val and its index
    %[a,b] = min(error_val);
    %a
    %b
    
    %Taking average errors
    %for r = 1:50
    %  for i = 1:size(X,1)-5
    %    index = rand(1)*size(X,1);
    %    newX = [newX; X(index)];
    %    newY = [newY; y(index)];
    %  end;
      
      %for i = 1:size(Xval,1)-5
    %  index = rand(1)*size(Xval,1);
    %    newXval = [newXval; Xval(index)];
    %    newyval = [newyval; yval(index)];
    %  end;
    
    
    %  for i = 1:length(lambda_vec)
    %  theta = trainLinearReg(newX,newy,lambda_vec(i,:));
    %  [J, grad] = linearRegCostFunction(newX, newy, theta, 0);
    %  [J1, grad1] = linearRegCostFunction(newXval, newyval, theta, 0);
      
    %  error_train(i,:) = (error_train(i,:) + J)/2;
    %  error_val(i,:) = (error_val(i,:) + J)/2;
    %  end;
    %endfor
    
end
