function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

  % Create New Figure
  figure; hold on;
  neg = find(y==0); pos = find(y==1);
  plot(X(pos,1),X(pos,2), 'k+', 'LineWidth', 2, 'Markersize', 7);
  plot(X(neg,1),X(neg,2), 'ko', 'LineWidth', 2, 'Markersize', 7);
  hold off;

end
