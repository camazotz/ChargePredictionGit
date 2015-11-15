# __author__ = 'nav'
#
# import csv
#
# with open('allPeptides.txt', 'rt', encoding='utf-8') as csvfile:
#     reader = csv.DictReader(csvfile, delimiter = '\t')
#     x = []
#     columnsOfInterest = ['Raw file','Type','Charge','m/z','Mass','Number of isotopic peaks','Retention time',
#                         'MS/MS IDs','Sequence','Length','Modifications','Intensity'];
#     #columnsOfInterest = ['test1','test2','test6','test7']
#     # for row in reader:
#     #     #for col in columnsOfInterest:
#     #     if row['test6'] != '':
#     #         x.extend(row[col] for col in columnsOfInterest)
#
#     desired_cols = (list(row[col] for col in columnsOfInterest) for row in reader if (row['Sequence'] != ' '
#                     and row['Modifications'] == 'Unmodified'))
#
#     writer = csv.writer(open('output.csv', 'w', encoding='utf-8'))
#     writer.writerows([columnsOfInterest])
#     writer.writerows(desired_cols)


# __author__ = 'nav'
#
# import csv
#
# with open('outputDuplicate.csv', 'r', encoding='utf-8') as csvfile:
#     reader = csv.DictReader(csvfile)
#     x = []
#     columnsOfInterest = ['Raw file','Type','Charge','m/z','Mass','Number of isotopic peaks','Retention time',
#                         'MS/MS IDs','Sequence','Length','Modifications','Intensity','FindDup'];
#     #columnsOfInterest = ['test1','test2','test6','test7']
#     # for row in reader:
#     #     #for col in columnsOfInterest:
#     #     if row['test6'] != '':
#     #         x.extend(row[col] for col in columnsOfInterest)
#
#     desired_cols = (list(row[col] for col in columnsOfInterest) for row in reader if (row['FindDup'] == 'TRUE'))
#
#     writer = csv.writer(open('outputPeptides.csv', 'w', encoding='utf-8'))
#     writer.writerows([columnsOfInterest])
#     writer.writerows(desired_cols)

#
# __author__ = 'nav'
#
# import csv
#
# with open('blahblah.csv', 'r', encoding='utf-8') as csvfile:
#     reader = csv.DictReader(csvfile)
#     columnsOfInterest = ['test1','test2','test3'] #,'test4','test5','test6','test7','test8'];
#     #columnsOfInterest = ['test1','test2','test6','test7']
#     # for row in reader:
#     #     #for col in columnsOfInterest:
#     #     if row['test6'] != '':
#     #         x.extend(row[col] for col in columnsOfInterest)
#
#     desired_cols = (list(row[col] for col in columnsOfInterest) for row in reader)# if (row['test6'] != ''))
#     ofile = open('testoutput.csv','w', encoding='utf-8');
#     writer = csv.writer(ofile)
#
#     writer.writerows([columnsOfInterest])
#     writer.writerows(desired_cols)