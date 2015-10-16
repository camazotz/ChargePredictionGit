__author__ = 'nav'

import pandas


columnsOfInterest = ['Raw file','Type','Charge','m/z','Mass','Number of isotopic peaks','Retention time',
                    'MS/MS IDs','Sequence','Length','Modifications','Intensity'];
data = pandas.read_csv('output.csv', names=columnsOfInterest, dtype=object)
sequenceList = list(data.Sequence);
rawFile = list(data['Raw file'])
s = set(sequenceList);
duplicatesList = [];
for x in sequenceList:
    if x in s:
        s.remove(x);
    else:
        duplicatesList.append(x);
    
print('hi');