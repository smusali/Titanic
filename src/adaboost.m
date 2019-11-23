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
classification_1 = fitensemble(train_data_1, train_label, 'AdaBoostM1', 300, 'Tree');
classification_2 = fitensemble(train_data_2, train_label, 'AdaBoostM1', 300, 'Tree');
cv_classificat_1 = crossval(classification_1);
cv_classificat_2 = crossval(classification_2);
fprintf('kfoldLoss of 1: %d;\n', kfoldLoss(cv_classificat_1));
fprintf('kfoldLoss of 2: %d;\n', kfoldLoss(cv_classificat_2));

prediction_1 = predict(classification_1, test_data_1);
prediction_2 = predict(classification_2, test_data_2);

submission_1 = [test_ID, prediction_1];
submission_2 = [test_ID, prediction_2];

csvwrite('../data/submission_e1.csv', submission_1);
csvwrite('../data/submission_e2.csv', submission_2);
