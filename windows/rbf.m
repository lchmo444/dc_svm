function [K] = rbf(X,X1,gamma)
%
%	K(i,j) = e^{-gamma*||x_i-x_j||^2}
%

n = size(X,1);
m = size(X1,1);
K = 2*X*X1';
%d = diag(K);
d = sum(X.^2,2);
d1 = sum(X1.^2,2);
K = K - d*ones(1,m) - ones(n,1)*d1';
K = K*gamma;
K = exp(K);
