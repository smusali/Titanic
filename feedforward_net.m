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
train_size = size(train_data_1, 1);
test_size = size(test_data_1, 1);
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
MAX_NUM_NODES = 5*min([size(train_data_1, 2), size(train_data_2, 2)]);
errData = zeros(MAX_NUM_NODES,2);
for numNodes=1:MAX_NUM_NODES
	fprintf('n=%d:\n', numNodes);
	net_1 = feedforwardnet(numNodes);
	net_1.trainParam.epochs = 100;
	net_1 = train(net_1, train_data_1', train_label');
	net_2 = feedforwardnet(numNodes);
	net_2.trainParam.epochs = 100;
	net_2 = train(net_2, train_data_2', train_label');
	yhat1 = net_1(train_data_1');
	yhat2 = net_2(train_data_2');
	for i=1:train_size
		if (yhat1(i) >= 0.5)
			yhat1(i) = 1;
		else
			yhat1(i) = 0;
		end
		if (yhat2(i) >= 0.5)
			yhat2(i) = 1;
		else
			yhat2(i) = 0;
		end
	end
	perf1 = perform(net_1, yhat1, train_label');
	perf2 = perform(net_2, yhat2, train_label');
	errData(numNodes, 1) = perf1;
	errData(numNodes, 2) = perf2;
	fprintf(' 1: %d;\n', perf1);
	fprintf(' 2: %d;\n', perf2);
end

plot(1:MAX_NUM_NODES, errData(:, 1), 'r+-', 1:MAX_NUM_NODES, errData(:, 2), 'g+-');

minerr = min(errData);
numNodes_1 = 0;
numNodes_2 = 0;
for i=1:MAX_NUM_NODES
	if (errData(i, 1) == minerr(1))
		numNodes_1 = i;
	end
	if (errData(i, 2) == minerr(2))
		numNodes_2 = i;
	end
end

net_1 = feedforwardnet(numNodes_1);
net_1.trainParam.epochs = 100;
net_1 = train(net_1, train_data_1', train_label');
net_2 = feedforwardnet(numNodes_2);
net_2.trainParam.epochs = 100;
net_2 = train(net_2, train_data_2', train_label');
out_1 = net_1(test_data_1');
out_2 = net_2(test_data_2');

for i=1:test_size
	if (out_1(i) >= 0.5)
		out_1(i) = 1;
	else
		out_1(i) = 0;
	end
	if (out_2(i) >= 0.5)
		out_2(i) = 1;
	else
		out_2(i) = 0;
	end
end

if (size(out_1, 2) ~= 1)
	out_1 = out_1';
	out_2 = out_2';
end

submission_1 = [test_ID, out_1];
submission_2 = [test_ID, out_2];

csvwrite('submission_ffnn1.csv', submission_1);
csvwrite('submission_ffnn2.csv', submission_2);
