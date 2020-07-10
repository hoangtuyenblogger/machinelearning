import csv
import pandas as pd
import statistics as st
import numpy as np
import random
import math
import sys





# Load data tu CSV file
def load_csv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


# splitdata
def split_data(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    9 + 6 + 999
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


# tinh toan gia tri trung binh cua moi thuoc tinh
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Tinh toan do lech chuan cho tung thuoc tinh
def std(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)

    return math.sqrt(variance)


# Chuyen ve cap du lieu  (Gia tri trung binh , do lech chuan)

def summarize(dataset):
    summaries = [(mean(attribute), standard_deviation(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]

    return summaries


def summarize_by_class(dataset):
    separated = separate_data(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)

    return summaries


# Tinh toan xac suat theo phan phoi Gause cua bien lien tuc the hien cac chi so suc khoe
def calculate_prob(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))

    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# Tinh xac suat cho moi chi so suc khoe theo class
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            print("x: ", x, "mean: ", mean, "stdev: ", stdev, "||", "summaries: ", summarize, "inputVector: ",
                  inputVector, "i:", [i])
            probabilities[classValue] *= calculate_prob(x, mean, stdev)

    return probabilities


# Du doan vector thuoc phan lop nao
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue

    return bestLabel


# Du doan tap du lieu testing thuoc vao phan lop nao
def get_predictions(summaries, test):
    predictions = []
    for i in range(len(test)):
        result = predict(summaries, test[i])
        predictions.append(result)

    return predictions


# Tinh toan do chinh xac cua phan lop
def get_accuracy(test, predictions):
    correct = 0
    for i in range(len(test)):
        if test[i][-1] == predictions[i]:
            correct += 1

    return (correct / float(len(test))) * 100.0


def main1():
    filename = 'hehe.csv'
    splitRatio = 0.7
    dataset = load_csv(filename)
    training, test = split_data(dataset, splitRatio)

    print('Data size {0} \nTraining Size={1} \nTest Size={2}'.format(len(dataset), len(training), len(test)))

    # chuẩn bị model
    summaries = summarize_by_class(training)

    # test model
    predictions = get_predictions(summaries, test)
    accuracy = get_accuracy(test, predictions)
    print('Accuracy of my implement: {0}%'.format(accuracy))


def size_training_test():
    filename = 'hehe.csv'
    splitRatio = 0.8
    dataset = load_csv(filename)
    training, test = split_data(dataset, splitRatio)

    print('Data size {0} \nTraining ={1} \nTest Size={2}'.format(len(dataset), len(training), len(test)))
################################## phần phía trên e lụm trên mạng nên đừng quan tâm :3
def main():
    size_training_test()
    filename = 'data.csv'
    data = pd.read_csv(filename)
    ## load dữ liệu vào mảng
    SV = data["SV"]
    NMCNTT = data["NMCNTT"]
    CSLT = data["CSLT"]
    KTLT = data["KTLT"]
    CTDLGT = data["CTDLGT"]
    MMT = data["MMT"]
    ####################################
    ndigit = 1
    print('Đối tượng thực hiện khảo sát chủ yếu là sinh viên năm',st.mode(SV))
    print('Điểm trung bình đối với từng môn học là:')
    print('\nMean_NMCNTT= {0} \nMean_CSLT= {1} \nMean_KTLT= {2} \nMean_CTDLGT= {3} \nMean_MMT={4}'.format(round(NMCNTT.mean(),ndigit), round(CSLT.mean(),ndigit), round(KTLT.mean(),ndigit), round(CTDLGT.mean(),ndigit), round(MMT.mean(),ndigit)))










if __name__ == '__main__':
    main()