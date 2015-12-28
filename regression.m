clc;
clear all;
close all;

% Age is the 3rd feature;
% Fare is the 7th feature.

test_data = csvread('testData.csv');
training_data = csvread('trainingData.csv');

Row_size = size(test_data, 1);
n = size(training_data, 1);
Column_size = size(test_data, 2);

Columns = []; numCols = 0;
Rows = []; numRows = 0;
specRow = 0;

for i=1:Row_size
	for j=1:Column_size
		if (test_data(i,j) == -1)
			if (j == 7)
				specRow = i;
			end
			if (~ismember(i, Rows))
				Rows = [Rows; i];
				numRows = numRows + 1;
			end
			if (~ismember(j, Columns))
				Columns = [Columns, j];
				numCols = numCols + 1;
			end
		end
	end
end

inRows = zeros(numRows-1,1);
cinRows = 0;
for i=1:numRows
	if (Rows(i) ~= specRow)
		cinRows = cinRows + 1;
		inRows(cinRows) = Rows(i);
	end
end

[trainAge_training_data, trainAge_training_label, trainAge_test_data] = divide_to(training_data, 3, Columns, [], Column_size, n);
[testAge_training_data, testAge_training_label, testAge_test_data] = divide_to(test_data, 3, Columns, inRows, Column_size, Row_size-cinRows);
Age_training_data = [trainAge_training_data; testAge_training_data];
Age_training_label = [trainAge_training_label; testAge_training_label];
Age_test_data = [trainAge_test_data; testAge_test_data];

Age_test_label = Age_test_data*inv(Age_training_data'*Age_training_data)*Age_training_data'*Age_training_label;
for i=1:cinRows
	p = Age_test_label(i);
	low = floor(p);
	high = ceil(p);
	med = (low+high)/2;
	if (p < low + 0.25)
		Age_test_label(i) = low;
	else
		if (p >= low + 0.25 & p < high - 0.25)
			Age_test_label(i) = med;
		else
			Age_test_label(i) = high;
		end
	end
end
test_data(inRows, 3) = Age_test_label;

[trainFare_training_data, trainFare_training_label, trainFare_test_data] = divide_to(training_data, 7, [7], [], Column_size, n);
[testFare_training_data, testFare_training_label, testFare_test_data] = divide_to(test_data, 7, [7], [specRow], Column_size, Row_size-1);
Fare_training_data = [trainFare_training_data; testFare_training_data];
Fare_training_label = [trainFare_training_label; testFare_training_label];
Fare_test_data = [trainFare_test_data; testFare_test_data];
Fare_test_label = Fare_test_data*inv(Fare_training_data'*Fare_training_data)*Fare_training_data'*Fare_training_label;
test_data(specRow, 7) = Fare_test_label;

csvwrite('test_data.csv', test_data);
csvwrite('training_data.csv', training_data);
