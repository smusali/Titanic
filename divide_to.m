function [training_data, training_label, test_data] = divide_to(data, labelColumn, Columns, Rows, Column_size, Row_size)
	reqRows = [];
	for i=1:Row_size
		if (~ismember(i, Rows))
			reqRows = [reqRows; i];
		end
	end
	reqColumns = [];
	for i=1:Column_size
		if (~ismember(i, Columns))
			reqColumns = [reqColumns, i];
		end
	end
	training_data = data(reqRows, reqColumns);
	test_data = data(Rows, reqColumns);
	training_label = data(reqRows, labelColumn);
end
