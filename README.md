# Titanic: Machine Learning from Disaster
Initially, we have `data/train/train.csv` and `data/test/test.csv`.

## Scripts:
1. `extraction.py`: this script takes `data/train/train.csv` and `data/test/test.csv` as input, and outputs:
	- `data/train/trainingData.csv`: train data matrix after some filtering and cleaning;
	- `data/test/testData.csv`: test data matrix after some filering and cleaning;
	- `data/train/trainingLabel.csv`: train class data, in which `1` shows the passenger `survived`;
	- `data/features.csv`: the feature names in the resulted data matrices;
	- `data/train/trainingID.csv`: IDs of the passengers given as train data;
	- `data/test/testID.csv`: IDs of the passengers given as test data. 

	Cleaning went in this way:
	- Completely dropped `Cabin` column because most data are not available;
	- Completely dropped the rows containing empty (`""`) info in train data matrix formation;
	- In test data matrix formation, used `-1` instead of empty (`""`) info;
	- For `Sex` and `Embarked` columns, encoding is straightforward;
	- `Ticket` data are also corrupted and old, but I tried to divide each ticket into alphabetical and numerical parts, and stored all those info from `data/train/train*` and `data/test/test*` data to some dictionary with finding some similarities and filtering in alphabetical parts, then encoded uniquely. 
	`NOTE: This part might ruin everything`.

2. `regression.m`: this script takes `data/test/testData.csv` and `data/train/trainingData.csv` as input, and outputs `data/test/test_data.csv` and `data/train/training_data.csv` with following fixing:
	- `data/train/training_data.csv` is the same as `data/train/trainingData.csv`;
	- In the `data/test/test_data.csv`, in only `3rd` (which is `Age`) and `7th` (which is `Fare`) columns we can see `-1` entries; so, we need to fix them. For that I used `divide_to` function which separates the data into its own training and test parts, then using valid entries in both `data/test/testData.csv` and `data/train/trainingData.csv`, by applying regular regression we can estimate missing values. So, placed the estimated values the place of missing ones.

3. `pca_analysis.m`: this script tries to find the best dimension to reduce the data using simple `Support Vector Machines` with `Cross Validation`. Here I used both {`data/train/trainingData.csv`, `data/test/testData.csv`} (in which the columns containing invalid entries are ignored) and {`data/train/training_data.m`, `data/test/test_data.m`} to see the effects.

Other parts are straightforward.

### NOTES:
- Ignore `data_selection.m`