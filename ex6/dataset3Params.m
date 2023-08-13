function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. 
%

  C = 1;
  sigma = 0.3;
  
  C_options = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
  sig_options = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
  %er = zeros(8,8); %C by sigma
  
  model = svmTrain(X, y, C_options(1), @(x1, x2) gaussianKernel(x1, x2, sigma(1)));
  predictions = svmPredict(model, Xval);
  prevEr = mean(double(predictions ~= yval));
  er = mean(double(predictions ~= yval));
  C = 0.01;
  sigma = 0.01;
  
  for cur_c = 1:length(C_options)
    for cur_sig = 1:length(sig_options)
      model = svmTrain(X, y, C_options(cur_c), @(x1, x2) gaussianKernel(x1, x2, sig_options(cur_sig)));
      predictions = svmPredict(model, Xval);
      if mean(double(predictions ~= yval)) < er
        er = mean(double(predictions ~= yval));
        C = C_options(cur_c);
        sigma = sig_options(cur_sig);
      %endif
      %Computes fraction that are incorrect
      %er(cur_c, cur_sig) = mean(double(predictions ~= yval));
    end;
end;










% =========================================================================

end
