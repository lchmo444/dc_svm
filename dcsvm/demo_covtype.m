addpath('../libsvm-3.14-nobias/matlab');

[y X] = libsvmread('../data/covtype.libsvm.binary.scale');
l = size(X,1);
p = randperm(l);
a = floor(l/5);
testX = X(p(1:a),:); testy = y(p(1:a));
trainX = X(p(a+1:end),:); trainy = y(p(a+1:end));

%% train/test rbf kernel SVM
ncluster = 64;
gamma = 32;
C = 32;
model = dcsvm_rbf_train(trainy, trainX, C, gamma, ncluster);
[labels accuracy] = dcsvm_test(testy, testX, model);
fprintf('RBF kernel, test accuracy %g\n', accuracy);

%% WARNING: polynomial training is slow
%% train/test polynomial kernel SVM
%{
ncluster = 64;
gamma = 8;
degree = 2;
C = 2;
model1 = dcsvm_poly_train(trainy, trainX, C, gamma, degree, ncluster);
[labels1 accuracy1] = dcsvm_test(testy, testX, model1);
fprintf('polynomial kernel, test accuracy %g\n', accuracy1);
%}
