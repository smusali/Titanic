clc;
clear all;
close all;

train_data_1 = csvread('../data/train/trainingData.csv');
test_data_1 = csvread('testData.csv');
req_Columns = [1, 2, 4, 5, 6, 8];
train_data_1 = train_data_1(:,req_Columns);
test_data_1 = test_data_1(:,req_Columns);
train_data_2 = csvread('../data/train/training_data.csv');
test_data_2 = csvread('test_data.csv');
test_ID = csvread('testID.csv');
train_size = size(train_data_1, 1);

train_label = csvread('../data/train/trainingLabel.csv');

train_data_1 = zscore(train_data_1);
train_data_2 = zscore(train_data_2);
test_data_1 = zscore(test_data_1);
test_data_2 = zscore(test_data_2);

MAX_P = min([size(train_data_1,2), size(train_data_2,2)]);
coeff_1 = pca(train_data_1);
coeff_2 = pca(train_data_2);
errData = zeros(MAX_P, 2);
for p=1:MAX_P
	fprintf('p=%d:\n', p);
	new_train_1 = train_data_1*coeff_1(:,1:p);
	new_train_2 = train_data_2*coeff_2(:,1:p);
	rng(1);
	SVMModel_1 = fitcsvm(new_train_1, train_label,'Standardize', true);
	SVMModel_2 = fitcsvm(new_train_2, train_label,'Standardize', true);
	CVSVMModel_1 = crossval(SVMModel_1);
	CVSVMModel_2 = crossval(SVMModel_2);
	errData(p, 1) = kfoldLoss(CVSVMModel_1);
	errData(p, 2) = kfoldLoss(CVSVMModel_2);
	fprintf(' 1: %d;\n', errData(p,1));
	fprintf(' 2: %d;\n', errData(p,2));
end

plot(1:MAX_P, errData(:, 1), 'r--', 1:MAX_P, errData(:, 2), 'g--');
