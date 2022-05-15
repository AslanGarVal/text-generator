from dataset_creator import DatasetFromCsv

dataset = DatasetFromCsv("../Data/examiner-date-text.csv")
dataset.import_data()

print(dataset.build_vocab())