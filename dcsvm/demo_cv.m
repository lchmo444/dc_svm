addpath('../libsvm-3.14-nobias/matlab');
maxNumCompThreads(1);

[trainy trainX] = libsvmread('../data/ijcnn1.train');
[testy, testX] = libsvmread('../data/ijcnn1.t');

ncluster = 10;

%% parameter selection
pars = dcsvm_cv_rbf(trainy, trainX, ncluster);
%% train/test rbf kernel SVM
gamma = pars.gamma;
C = pars.C;
fprintf('Best parameter: C=%g, gamma=%g\n', C, gamma);
fprintf('Start training Gaussian kernel SVM with early prediction\n', ncluster);
timebegin = cputime;
model = dcsvm_rbf_train(trainy, trainX, C, gamma, ncluster);
trainingtime = cputime - timebegin;
[labels accuracy] = dcsvm_test(testy, testX, model);
fprintf('RBF kernel, DCSVM-early test accuracy %g, training time %g seconds\n', accuracy, trainingtime);

