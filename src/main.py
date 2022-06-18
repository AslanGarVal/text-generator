import torch

from dataset_creator import DatasetFromCsv
from train_model import LSTM, Model

dataset = DatasetFromCsv("../Data/examiner-date-text.csv")
dataset.import_data()

print(dataset.build_vocab())

lstm = LSTM(10, 10)

model = Model(10, 10, 10, 8)

x = torch.zeros(10)
h, c = lstm.init_hidden_and_cell_state()

print(x)
print(lstm.forward_pass(x, h, c))
print(model.forward_pass(x, h, c))