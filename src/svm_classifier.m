clc;
clear all;
close all;

train_data_1 = csvread('../data/train/trainingData.csv');
test_data_1 = csvread('../data/test/testData.csv');
req_Columns = [1, 2, 4, 5, 6, 8];
train_data_1 = train_data_1(:,req_Columns);
test_data_1 = test_data_1(:,req_Columns);
train_data_2 = csvread('../data/train/training_data.csv');
test_data_2 = csvread('../data/test/test_data.csv');
test_ID = csvread('../data/test/testID.csv');
train_size = size(train_data_1, 1);
test_size = size(test_data_1, 1);
train_label = csvread('../data/train/trainingLabel.csv');


train_data_1 = zscore(train_data_1);
train_data_2 = zscore(train_data_2);
test_data_1 = zscore(test_data_1);
test_data_2 = zscore(test_data_2);

coeff_1 = pca(train_data_1);
coeff_2 = pca(train_data_2);
p = 5;

train_data_1 = train_data_1*coeff_1(:,1:p);
train_data_2 = train_data_2*coeff_2(:,1:p);
test_data_1 = test_data_1*coeff_1(:,1:p);
test_data_2 = test_data_2*coeff_2(:,1:p);

rng(1);
SVMModel_1 = fitcsvm(train_data_1, train_label, 'ClassNames', [0, 1], 'KernelFunction', 'rbf', 'Standardize', true, 'KernelScale', 'auto');
SVMModel_2 = fitcsvm(train_data_2, train_label, 'ClassNames', [0, 1], 'KernelFunction', 'rbf', 'Standardize', true, 'KernelScale', 'auto');
CVSVMModel_1 = crossval(SVMModel_1);
CVSVMModel_2 = crossval(SVMModel_2);
fprintf('1: %d;\n', kfoldLoss(CVSVMModel_1));
fprintf('2: %d;\n', kfoldLoss(CVSVMModel_2));
out_1 = predict(SVMModel_1, test_data_1);
out_2 = predict(SVMModel_2, test_data_2);

submission_1 = [test_ID, out_1];
submission_2 = [test_ID, out_2];

csvwrite('../data/submission_svm_1.csv', submission_1);
csvwrite('../data/submission_svm_2.csv', submission_2);
