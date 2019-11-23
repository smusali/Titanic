import csv, numpy as np
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def is_okay(row, k):
    it = 0;
    for i in row:
        if (it == k):
            it += 1;
            continue;
        if (i == ''):
            return 0;
        else:
            it += 1;
    return 1;

print 'Reading has started';
## Reading the given data:
header = [];
trainingData = [];
testData = [];
with open('./data/train/train.csv', 'rb') as trainFile:
    rowNum = 0;
    givenTrainData = csv.reader(trainFile, delimiter=',');
    for row in givenTrainData:
        if (rowNum == 0):
            header = row;
            rowNum += 1;
        else:
            if (is_okay(row, 10)):
                trainingData.append(row);

with open('./data/test/test.csv', 'rb') as testFile:
    rowNum = 0;
    givenTestData = csv.reader(testFile, delimiter=',');
    for row in givenTestData:
        if (rowNum == 0):
            rowNum += 1;
        else:
            testData.append(row);
print 'Reading has finished';
##['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 
## 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

##['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']

trainingLabel = []; 
trainingPclass = []; testPclass = [];
trainingSex = []; testSex = [];
trainingAge = []; testAge = [];
trainingSibSp = []; testSibSp = [];
trainingParch = []; testParch = [];
trainingTicket = []; testTicket = [];
trainingFare = []; testFare = [];
trainingEmbarked = []; testEmbarked = [];
trainingId = []; testId = [];

print 'Separating data has started';
## Divide and place the data into appropriate places:
for row in trainingData:
    trainingId.append(int(row[0]));
    trainingLabel.append(int(row[1]));
    trainingPclass.append(row[2]);
    trainingSex.append(row[4]);
    trainingAge.append(row[5]);
    trainingSibSp.append(row[6]);
    trainingParch.append(row[7]);
    trainingTicket.append(row[8]);
    trainingFare.append(row[9]);
    trainingEmbarked.append(row[11]);

extractedHeader = [];
extractedHeader.append(header[2]);
extractedHeader.append(header[4]);
extractedHeader.append(header[5]);
extractedHeader.append(header[6]);
extractedHeader.append(header[7]);
extractedHeader.append(header[8]);
extractedHeader.append(header[9]);
extractedHeader.append(header[11]);

for row in testData:
    testId.append(int(row[0]));
    testPclass.append(row[1]);
    testSex.append(row[3]);
    testAge.append(row[4]);
    testSibSp.append(row[5]);
    testParch.append(row[6]);
    testTicket.append(row[7]);
    testFare.append(row[8]);
    testEmbarked.append(row[10]);
print 'Separating data has finished';

trainingNumber = np.size(trainingPclass);
testNumber = np.size(testPclass);
print 'Training size is '+str(trainingNumber);
print 'Test size is '+str(testNumber);

print 'Numerization has started';
## Trying to make everything numeric:
numTrainingData = []; numTestData = [];
numTrainingPclass = []; numTestPclass = [];
numTrainingSex = []; numTestSex = [];
numTrainingAge = []; numTestAge = [];
numTrainingSibSp = []; numTestSibSp = [];
numTrainingParch = []; numTestParch = [];
numTrainingTicket = []; numTestTicket = [];
numTrainingFare = []; numTestFare = [];
numTrainingEmbarked = []; numTestEmbarked = [];

for i in xrange(trainingNumber):
    Pclass = trainingPclass[i];
    Sex = trainingSex[i];
    Age = trainingAge[i];
    SibSp = trainingSibSp[i];
    Parch = trainingParch[i];
    Fare = trainingFare[i];
    Embarked = trainingEmbarked[i];
    if (Pclass == ''):
        print " Training Data "+str(i+1)+" Pclass is missing";
        numTrainingPclass.append(-1);
    else:
        numTrainingPclass.append(int(Pclass));
    if (Sex == 'female'):
        numTrainingSex.append(0);
    elif (Sex == 'male'):
        numTrainingSex.append(1);
    else:
        print " Training Data "+str(i+1)+" Sex is missing";
        numTrainingSex.append(-1);
    if (Age == ''):
        print " Training Data "+str(i+1)+" Age is missing";
        numTrainingAge.append(-1);
    else:
        numTrainingAge.append(float(Age));
    if (SibSp == ''):
        print " Training Data "+str(i+1)+" SibSp is missing";
        numTrainingSibSp.append(-1);
    else:
        numTrainingSibSp.append(int(SibSp));
    if (Parch == ''):
        print " Training Data "+str(i+1)+" Parch is missing";
        numTrainingParch.append(-1);
    else:
        numTrainingParch.append(int(Parch));
    if (Fare == ''):
        print " Training Data "+str(i+1)+" Fare is missing";
        numTrainingFare.append(-1);
    else:
        numTrainingFare.append(float(Fare));
    if (Embarked == 'S'):
        numTrainingEmbarked.append(0);
    elif (Embarked == 'C'):
        numTrainingEmbarked.append(1);
    elif (Embarked == 'Q'):
        numTrainingEmbarked.append(2);
    else:
        print " Training Data "+str(i+1)+" Embarked is missing";
        numTrainingEmbarked.append(-1);
        
for i in xrange(testNumber):
    Pclass = testPclass[i];
    Sex = testSex[i];
    Age = testAge[i];
    SibSp = testSibSp[i];
    Parch = testParch[i];
    Fare = testFare[i];
    Embarked = testEmbarked[i];
    if (Pclass == ''):
        print " Test Data "+str(i+1)+" Pclass is missing";
        numTestPclass.append(-1);
    else:
        numTestPclass.append(int(Pclass));
    if (Sex == 'female'):
        numTestSex.append(0);
    elif (Sex == 'male'):
        numTestSex.append(1);
    else:
        print " Test Data "+str(i+1)+" Sex is missing";
        numTestSex.append(-1);
    if (Age == ''):
        print " Test Data "+str(i+1)+" Age is missing";
        numTestAge.append(-1);
    else:
        numTestAge.append(float(Age));
    if (SibSp == ''):
        print " Test Data "+str(i+1)+" SibSp is missing";
        numTestSibSp.append(-1);
    else:
        numTestSibSp.append(int(SibSp));
    if (Parch == ''):
        print " Test Data "+str(i+1)+" Parch is missing";
        numTestParch.append(-1);
    else:
        numTestParch.append(int(Parch));
    if (Fare == ''):
        print " Test Data "+str(i+1)+" Fare is missing";
        numTestFare.append(-1);
    else:
        numTestFare.append(float(Fare));
    if (Embarked == 'S'):
        numTestEmbarked.append(0);
    elif (Embarked == 'C'):
        numTestEmbarked.append(1);
    elif (Embarked == 'Q'):
        numTestEmbarked.append(2);
    else:
        print " Test Data "+str(i+1)+" Embarked is missing";
        numTestEmbarked.append(-1);
print 'Numerization has partially finished';
print 'Numerizating Tickets has started';
## The hardest part it convert ticket to numeric value, since
## it seems little bit old information.
TrainingLetter = [];
TrainingNumer = [];
TestLetter = [];
TestNumer = [];
UniqueLetter = [];
for ticket in trainingTicket:
    twoPartTicket = ticket.split(' ');
    L = len(twoPartTicket);
    if (L == 1):
        if (twoPartTicket[0] == 'LINE'):
            TrainingLetter.append('LINE'.lower());
            TrainingNumer.append(0);
        else:
            TrainingLetter.append('-'.lower());
            TrainingNumer.append(int(twoPartTicket[0]));
    else:
        letter = '';
        for i in xrange(L-1):
            letter = letter + twoPartTicket[i];
        numer = twoPartTicket[L-1];
        letter = letter.replace('/','');
        letter = letter.replace('.','');
        TrainingLetter.append(letter.lower());
        TrainingNumer.append(int(numer));

for i in xrange(trainingNumber):
    first = TrainingLetter[i];
    for j in xrange(trainingNumber-i):
        second = TrainingLetter[j+i];
        if (similar(first, second) >= 0.8):
            TrainingLetter[j+i] = first;

for ticket in testTicket:
    twoPartTicket = ticket.split(' ');
    L = len(twoPartTicket);
    if (L == 1):
        if (twoPartTicket[0] == 'LINE'):
            TestLetter.append('LINE'.lower());
            TestNumer.append(0);
        else:
            TestLetter.append('-'.lower());
            TestNumer.append(int(twoPartTicket[0]));
    else:
        letter = '';
        for i in xrange(L-1):
            letter = letter + twoPartTicket[i];
        numer = twoPartTicket[L-1];
        letter = letter.replace('/','');
        letter = letter.replace('.','');
        TestLetter.append(letter.lower());
        TestNumer.append(int(numer));

for i in xrange(testNumber):
    first = TestLetter[i];
    for j in xrange(testNumber-i):
        second = TestLetter[j+i];
        if (similar(first, second) >= 0.8):
            TestLetter[j+i] = first;

UniqueLetter = list(np.unique(TrainingLetter));
UniqueLetter.extend(np.unique(TestLetter));
UniqueLetter = np.unique(UniqueLetter);
maxNumer = max([max(TrainingNumer), max(TestNumer)]);

for i in xrange(trainingNumber):
    letter = TrainingLetter[i];
    numer = TrainingNumer[i];
    letterIndex = np.where(UniqueLetter == letter)[0][0];
    numTrainingTicket.append(letterIndex*maxNumer+numer);
    
for i in xrange(testNumber):
    letter = TestLetter[i];
    numer = TestNumer[i];
    letterIndex = np.where(UniqueLetter == letter)[0][0];
    numTestTicket.append(letterIndex*maxNumer+numer);
print 'Numerizating Tickets has finished';
print 'Merging and forming data have started';
##['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
## Almost done:
resultedTrainingData = []; resultedTestData = [];

resultedTrainingData.append(np.array(numTrainingPclass));
resultedTrainingData.append(np.array(numTrainingSex));
resultedTrainingData.append(np.array(numTrainingAge));
resultedTrainingData.append(np.array(numTrainingSibSp));
resultedTrainingData.append(np.array(numTrainingParch));
resultedTrainingData.append(np.array(numTrainingTicket));
resultedTrainingData.append(np.array(numTrainingFare));
resultedTrainingData.append(np.array(numTrainingEmbarked));
resultedTrainingData = np.transpose(np.array(resultedTrainingData));

resultedTestData.append(np.array(numTestPclass));
resultedTestData.append(np.array(numTestSex));
resultedTestData.append(np.array(numTestAge));
resultedTestData.append(np.array(numTestSibSp));
resultedTestData.append(np.array(numTestParch));
resultedTestData.append(np.array(numTestTicket));
resultedTestData.append(np.array(numTestFare));
resultedTestData.append(np.array(numTestEmbarked));
resultedTestData = np.transpose(np.array(resultedTestData));
print 'Merging and forming data have finished';
print 'Saving has started';
## Data are ready to be used
np.savetxt('./data/train/trainingData.csv', resultedTrainingData, delimiter=',');
np.savetxt('./data/test/testData.csv', resultedTestData, delimiter=',');
np.savetxt('./data/train/trainingLabel.csv', trainingLabel, delimiter=',');
np.savetxt('./data/features.csv', extractedHeader, delimiter=',', fmt='%s');
np.savetxt('./data/train/trainingID.csv', trainingId, delimiter=',');
np.savetxt('./data/test/testID.csv', testId, delimiter=',');
print 'Saving has finished';
print 'Thanks for attention =)';