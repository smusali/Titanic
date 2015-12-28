# Kaggle challenge "Titanic: Machine Learning from Disaster"

Initially, we have *train.csv* and *test.csv*.

## Scripts:
1. *extraction.py*: this script takes *train.csv* and *test.csv* as input, and outputs:
	- *trainingData.csv*: train data matrix after some filtering and cleaning;
	- *testData.csv*: test data matrix after some filering and cleaning;
	- *trainingLabel.csv*: train class data, in which **1** shows the passenger **survived**;
	- *Features.csv*: the feature names in the resulted data matrices;
	- *trainingID.csv*: IDs of the passengers given as train data;
	- *testID.csv*: IDs of the passengers given as test data. 

	Cleaning went in this way:
	- Completely dropped **Cabin** column because most data are not available;
	- Completely dropped the rows containing empty (*""*) info in train data matrix formation;
	- In test data matrix formation, used *-1* instead of empty (*""*) info;
	- For **Sex** and **Embarked** columns, encoding is straightforward;
	- **Ticket** data are also corrupted and old, but I tried to divide each ticket into alphabetical and numerical parts, and stored all those info from *train* and *test* data to some dictionary with finding some similarities and filtering in alphabetical parts, then encoded uniquely. 
	*NOTE: This part might ruin everything*.

2. *regression.m*: this script takes *testData.csv* and *trainingData.csv* as input, and outputs *test_data.csv* and *training_data.csv* with following fixing:
	- *training_data.csv* is the same as *trainingData.csv*;
	- In the *test_data.csv*, in only **3rd** (which is **Age**) and **7th** (which is **Fare**) columns we can see *-1* entries; so, we need to fix them. For that I used *divide_to* function which separates the data into its own training and test parts, then using valid entries in both *testData.m* and *trainingData.m*, by applying regular regression we can estimate missing values. So, placed the estimated values the place of missing ones.

3. *pca_analysis.m*: this script tries to find the best dimension to reduce the data using simple *Support Vector Machines* with *Cross Validation*. Here I used both {*trainingData.m*, *testData.m*} (in which the columns containing invalid entries are ignored) and {*training_data.m*, *test_data.m*} to see the effects.

Others are straightforward.
*NOTE: Ignore data_selection.m*

*P.S.: Still trying to get **zero true error**, since in the ranking some users got that*
