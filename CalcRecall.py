import cPickle
import numpy as np

cpDir0 = 'D:/PythonProjects/EvalDetection/recall_all_label.pkl'
cpDir1 = 'D:/PythonProjects/EvalDetection/recall_train1.pkl'
cpDir2 = 'D:/PythonProjects/EvalDetection/recall_train2.pkl'

recall = cPickle.load(open(cpDir2))

sumobj = 0
sumtp = 0
for i in range(len(recall)):
    sumobj += recall[i][0]
    sumtp += recall[i][1]

print sumobj, sumtp
print 'recall rate:', sumtp/sumobj

# recall = cPickle.load(open(cpDir1))
#
# for i in range(len(recall)):
#     sumobj += recall[i][0]
#     sumtp += recall[i][1]
#
# print sumobj, sumtp
# print 'recall rate:', sumtp/sumobj
#
# recall = cPickle.load(open(cpDir2))
#
# for i in range(len(recall)):
#     sumobj += recall[i][0]
#     sumtp += recall[i][1]
#
# print sumobj, sumtp
# print 'recall rate:', sumtp/sumobj
