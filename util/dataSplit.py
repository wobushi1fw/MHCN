class DataSplit(object):

    def __init__(self):
        pass

    @staticmethod
    def crossValidation(data, k, output=False, path='./', order=1, binarized=False):
        if k<=1 or k>10:
            k=3
        for i in range(k):
            trainingSet = []
            testSet = []
            for ind,line in enumerate(data):
                if ind%k == i:
                    if binarized:
                        if line[2]:
                            testSet.append(line[:])
                    else:
                        testSet.append(line[:])
                else:
                    trainingSet.append(line[:])
            yield trainingSet,testSet


