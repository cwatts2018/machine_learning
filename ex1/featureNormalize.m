function [X_norm, mu, sigma] = featureNormalize(X)
%   FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is a good preprocessing step to do when
%   working with learning algorithms.

% Set values
X_norm = X;
mu = zeros(1, size(X, 2)); %the 2 refers to the # of rows
sigma = zeros(1, size(X, 2));

mu = mean(X);
sigma = std(X);
[r,c] = size(X);
for i = 1:c  
  X_norm(:,i) =(X(:,i)- mu(1,i))/sigma(1,i);
  end 
end
