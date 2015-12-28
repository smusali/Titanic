clc;
clear all;
close all;

train_data_1 = csvread('trainingData.csv');
test_data_1 = csvread('testData.csv');
req_Columns = [1, 2, 4, 5, 6, 8];
train_data_1 = train_data_1(:,req_Columns);
test_data_1 = test_data_1(:,req_Columns);
train_data_2 = csvread('training_data.csv');
test_data_2 = csvread('test_data.csv');
test_ID = csvread('testID.csv');

train_label = csvread('trainingLabel.csv');


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
MAX_NUM_TREES = 5;
errData = zeros(MAX_NUM_TREES,2);
for i=1:MAX_NUM_TREES
	fprintf('n=%d:\n', i);
	tree_1 = TreeBagger(100*i, train_data_1, train_label, 'OOBPrediction', 'on', 'Method', 'classification');
	tree_2 = TreeBagger(100*i, train_data_2, train_label, 'OOBPrediction', 'on', 'Method', 'classification');
	errData(i, 1) = mean(oobError(tree_1));
	errData(i, 2) = mean(oobError(tree_2));
	fprintf(' 1: %d;\n', errData(i, 1));
	fprintf(' 2: %d;\n', errData(i, 2));
	%oobError(tree_1)
	%oobError(tree_2)
end

plot(1:MAX_NUM_TREES, errData(:, 1), 'r+-', 1:MAX_NUM_TREES, errData(:, 2), 'g+-');
