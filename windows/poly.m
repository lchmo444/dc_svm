function [K] = poly(X,X1,gamma, degree)
%
%	K(i,j) = (x_i'*x_j*gamma + 1)^degree
%

n = size(X,1);
m = size(X1,1);
K = gamma*X*X1'+1;
K = K.^degree;
