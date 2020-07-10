
import csv
import pandas
def load_data(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]

    return dataset


filename = 'tieu_duong.csv'
data = pandas.read_csv(filename)
print(data)