__author__ = 'nav'
import numpy as np
import csv
import math
import pickle
from collections import OrderedDict
from sklearn import linear_model
from sklearn import cross_validation
import matplotlib.pyplot as plt

THRESHOLD_LEVEL = 0.5
# X = residues, basic res = Arg(R), Lys(K), His(H)

def nCr(n,k):
    fact = math.factorial
    return fact(n) / (fact(k)*fact(n-k))


def MStep(peptides, basicCount):
    numSuccess = 0.0
    numTrials = 0.0
    #numPeptides = 0
    for record in peptides:
        numSuccess = numSuccess + float(record[1]) * float(record[0])
        numTrials = numTrials + float(record[1]) * basicCount
    # for record in peptides:
    #     numSuccess = numSuccess + int(record[3]) * int(record[1])
    #     numTrials = numTrials + int(record[3]) * basicCount
    #     numPeptides = numPeptides + int(record[3])
    maxLikelihood = numSuccess / numTrials
    #return (maxLikelihood,numPeptides)
    return maxLikelihood


def EStep(basicCount, maxLikelihood, numPeptides, peptideList):
    predictedPeptides = [1,2]
    threshold = 0#float("-inf")

    for i in range(0,basicCount+1):
        iProb = nCr(basicCount,i) * math.pow(maxLikelihood,i) \
        * math.pow(1-maxLikelihood, basicCount-i)
        #print(peptideList[0][2], ' probability of +',i,' ',iProb)
        predictedVal = iProb * numPeptides
        #print ('# +',i, '= ',predictedVal)
        predictedPeptides = np.vstack((predictedPeptides,[i,predictedVal]))
        actualVal = peptideList[np.where(peptideList[:,1].astype(int) == i)]

        if actualVal.size != 0:
            if (abs(actualVal[0][3].astype(int)-predictedVal) > THRESHOLD_LEVEL):
                threshold = threshold + abs(actualVal[0][3].astype(int)-predictedVal)
                #if threshold > THRESHOLD_LEVEL:
                predictedPeptides[predictedPeptides.shape[0]-1][predictedPeptides.shape[1]-1] \
                = actualVal[0][3].astype(float)

    predictedPeptides = np.delete(predictedPeptides,(0),axis=0)
    return(predictedPeptides, threshold)


with open('sortedOutputPeptides.csv', 'rt', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    columnsOfInterest = ['Raw file','Charge','Sequence','Intensity']
    #data = np.array(['Raw file','Charge','Sequence','Intensity'])

    desired_cols = (list(row[col] for col in columnsOfInterest) for row in reader)
    #print(', '.join(map(str,desired_cols)))
    print('start reading')
    # for row in reader:
    #     newrow = list(row[col] for col in columnsOfInterest)
    #     data = np.vstack([data, newrow])

    #data = np.zeros(len(list(desired_cols)))
    # for i, el in enumerate(desired_cols):
    #     data[i] = el

    data = np.array(list(desired_cols))

    #ata = np.asarray(desired_cols)
    #print(data)
    # Basic residues are Arg,Lys,His
    print('Done reading')
    keyValues = [('A',0), ('R',0), ('N',0), ('D',0), ('C',0), ('Q',0), ('E',0),
             ('G',0), ('H',0), ('I',0), ('L',0), ('K',0), ('M',0), ('F',0),
             ('P',0), ('S',0), ('T',0), ('W',0), ('Y',0), ('V',0), ('B',0),
             ('Z',0), ('X',0)]

    numNotConverged = 0
    XElements = np.zeros(23)
    yColumn = np.array([])
    pSequence = ''
    peptides = np.array([])
    for idx, row1 in enumerate(data):
        if row1[2] == 'Sequence':
            continue
        if idx == len(data)-1:
            peptides = np.vstack([peptides,row1])
        if pSequence != row1[2] or (idx == len(data) -1):
            if peptides.size != 0:
                #print(peptides)

                XMatrix = OrderedDict(keyValues)
                for ch in peptides[0][2]:
                    if ch in XMatrix:
                        XMatrix[ch] = XMatrix[ch] +1

                numPeptides = np.sum(peptides[:,3].astype(int))
                #basicCount = peptides[0][2].count('A')
                basicCount = XMatrix['R'] + XMatrix['K'] + XMatrix['H']

                observedMax = int(max(peptides[:,1]))
                if observedMax > basicCount:
                    basicCount = observedMax
                #print(basicCount)

                threshold = math.inf
                maxLikelihood = 0
                #print(basicCount)
                #maxLikelihood, numPeptides = MStep(peptides,basicCount)[0:2]
                peptideColumns = np.column_stack((peptides[:,1],peptides[:,3]))

                predictions = []
                numIter = 0
                while threshold > THRESHOLD_LEVEL and numIter < 50:
                    # M Step
                    maxLikelihood = MStep(peptideColumns,basicCount)
                    #print(maxLikelihood,numPeptides)
                    #print(maxLikelihood)
                    # E Step
                    predictions, threshold = EStep(basicCount,maxLikelihood,numPeptides,peptides)[0:2]
                    peptideColumns = predictions
                    #print(peptideColumns)
                    numPeptides = peptideColumns[:, 1].sum()
                    numIter += 1

                # print(predictions)
                #print(maxLikelihood)
                if maxLikelihood != 1.0:
                    if numIter == 50:
                        numNotConverged += 1
                    yColumn = np.append(yColumn,maxLikelihood)
                    XArray = np.array([v for v in XMatrix.values()])
                    #print(XElements)
                    #print('XAr: ',XArray)
                    XElements = np.vstack([XElements,XArray])
                    # print(peptides[:,1])

            pSequence = row1[2]
            peptides = row1
        else:
            peptides = np.vstack([peptides,row1])

    XElements = np.delete(XElements,0,0)
    print(numNotConverged)
    #print(XElements.shape)
    #print(yColumn.size)
    #print(yColumn)
    #print(XElements)

    with open('X_Elements.pkl','wb') as fid:
        pickle.dump(XElements,fid)

    with open('Y_Column.pkl','wb') as fid:
        pickle.dump(yColumn,fid)

    with open('X_Elements.pkl','rb') as fid:
        X_Elements = pickle.load(fid)

    with open('Y_Column.pkl','rb') as fid:
        yColumn = pickle.load(fid)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(XElements,
                                                                         yColumn,
                                                                         test_size=0.4,
                                                                         random_state = 0)
    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)

    with open ('my_dumped_linreg.pkl','wb') as fid:
        pickle.dump(clf, fid)

    with open('my_dumped_linreg.pkl','rb') as fid:
        clf = pickle.load(fid)

    print(clf.coef_)
    print('\n')
    print(clf.score(X_test,y_test))

    #plt.scatter(X_test, y_test)
    # plt.plot(X_test, clf.predict(X_test), color='blue', linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    #
    # plt.show()
    #print(clf.coef_)
    #check row['sequence'].find('A') != -1 && ...