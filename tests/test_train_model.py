from unittest import TestCase

import torch

from train_model import sentence_to_tensor, LSTM, Model


class Test(TestCase):
    def test_sentence_to_tensor(self):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        eric_tensor = torch.zeros(4, 1, 26)
        # e = 5, r = 18, i = 9, c = 3
        for (letter, index) in zip(range(4), [4, 17, 8, 2]):
            eric_tensor[letter][0][index] = 1
        self.assertEqual(torch.equal(sentence_to_tensor('a', 'a'), torch.Tensor([[[1]]])), True)
        self.assertEqual(torch.equal(sentence_to_tensor('ab', 'abcd'), torch.Tensor([[[1., 0., 0., 0.]],
                                                                                     [[0., 1., 0., 0.]]])), True)
        self.assertEqual(torch.equal(sentence_to_tensor('eric', letters), eric_tensor), True)


class TestLSTM(TestCase):
    def test_init_hidden_and_cell_state(self):
        hidden_size = 32
        input_size = 10

        lstm = LSTM(input_size, hidden_size)
        hidden, cell_state = lstm.init_hidden_and_cell_state()

        self.assertEqual(
            torch.equal(
                hidden,
                torch.zeros(1, hidden_size)
            ),
            True
        )

        self.assertEqual(torch.equal(cell_state, torch.zeros(1, hidden_size)), True)

    def test_forward_pass_returns_three_elements(self):
        hidden_size = 32
        input_size = 10

        lstm = LSTM(input_size, hidden_size)
        mock_input = torch.zeros(1, input_size)
        hidden, cell_state = lstm.init_hidden_and_cell_state()

        self.assertEqual(len(lstm.forward_pass(mock_input, hidden, cell_state)), 3, True)


class TestModel(TestCase):
    def test_forward_pass(self):
        final_output_size = 27
        input_size = final_output_size
        hidden_lstm_size = 64
        hidden_fc_size = 128
        batch_size = 10

        model = Model(final_output_size, input_size, hidden_lstm_size, hidden_fc_size)

        mock_input = torch.zeros(batch_size, 1, input_size)
        hidden, cell_state = model.lstm_unit.init_hidden_and_cell_state()

        # we get three outputs on each forward run
        self.assertEqual(len(model.forward_pass(mock_input, hidden, cell_state)), 3)
        # softmax produces a row wise sum of 1.0
        self.assertEqual(
            torch.equal(
                torch.sum(model.forward_pass(mock_input, hidden, cell_state)[0], -1),
                torch.ones(batch_size, 1)
            ),
            True
        )
