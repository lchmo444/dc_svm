addpath('libsvm_windows');
maxNumCompThreads(1);

[trainy trainX] = libsvmread('../data/ijcnn1.train');
[testy, testX] = libsvmread('../data/ijcnn1.t');
%% train/test rbf kernel SVM
ncluster = 10;
gamma = 2;
C = 32;
fprintf('Start training Gaussian kernel SVM with early prediction\n', ncluster);
timebegin = cputime;
model = dcsvm_rbf_train(trainy, trainX, C, gamma, ncluster);
trainingtime = cputime - timebegin;
[labels accuracy] = dcsvm_test(testy, testX, model);
fprintf('RBF kernel, DCSVM-early test accuracy %g, training time %g seconds\n', accuracy, trainingtime);
