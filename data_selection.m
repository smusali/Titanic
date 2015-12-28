clc;
clear all;
close all;

%% Here I am trying to see which size and data are suitable:

trainData 	= csvread('trainData.csv');
trainData	= zscore(trainData);
trainBinData 	= csvread('trainBinData.csv');
trainBinData	= zscore(trainBinData);
trainFreqData 	= csvread('trainFreqData.csv');
trainFreqData	= zscore(trainFreqData);
trainLabel 	= csvread('trainLabel.csv');
fprintf('Data has been loaded\n');

trainSize 	= size(trainData, 1);
P_MAX 		= size(trainData, 2);
for i=1:trainSize
	trainLabel(i) = (trainLabel(i)+1)/2;
end

coeff 		= pca(trainData);
coeffBin 	= pca(trainBinData);
coeffFreq 	= pca(trainFreqData);
fprintf('Coefficients are ready\n');

errData 	= zeros(P_MAX, 3);

for p=1:P_MAX
	fprintf('p=%d:\n', p);
	newTrainData	= trainData*coeff(:,1:p);
	newTrainBinData = trainBinData*coeffBin(:,1:p);
	newTrainFreqData= trainFreqData*coeffFreq(:,1:p);

	net		= feedforwardnet(1);
	netBin		= feedforwardnet(1);
	netFreq		= feedforwardnet(1);
	
	net		= train(net, newTrainData', trainLabel');
	netBin		= train(netBin, newTrainBinData', trainLabel');
	netFreq		= train(netFreq, newTrainFreqData', trainLabel');

	y		= net(newTrainData');
	yBin		= net(newTrainBinData');
	yFreq		= net(newTrainFreqData');
	
	errData(p, 1)	= perform(net, y, trainLabel');
	errData(p, 2)	= perform(netBin, yBin, trainLabel');
	errData(p, 3)	= perform(netFreq, yFreq, trainLabel');

	fprintf(' 1: %d;\n', errData(p,1));
	fprintf(' 2: %d;\n', errData(p,2));
	fprintf(' 3: %d;\n', errData(p,3));
end
