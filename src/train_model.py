import torch
import torch.nn as nn


def sentence_to_tensor(sentence: str, vocab: str) -> torch.Tensor:
    """
    Takes a sentence string and a vocabulary to one-hot-encode each character of the sentence
    return the sentence as a tensor containing the one-hot-encoded vectors

    :param sentence: string
    :param vocab: string: all the characters to be used
    :return: a torch.Tensor of size (len(sentence), 1, len(vocab))
    """
    tensor = torch.zeros(len(sentence), 1, len(vocab))
    for c_index, c in enumerate(sentence):
        vocab_index = vocab.find(c)
        tensor[c_index][0][vocab_index] = 1
    return tensor


# TODO: Implement RNN

class LSTM(nn.Module):
    """
    This class implements a simple LSTM layer in pytorch
    """

    def __init__(self, input_size, hidden_size):
        """
        Constructor for the LSTM layer

        :param input_size: dimension of the input size
        :param hidden_size: dimension of the output, hidden, and cell state sizes
        """
        super(LSTM, self).__init__()

        # instantiate dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size

        # instantiate weights (forget gate, update gate, output gate, cell input gate)
        self.input_to_forget = nn.Linear(input_size, hidden_size)
        self.hidden_to_forget = nn.Linear(hidden_size, hidden_size)

        self.input_to_update = nn.Linear(input_size, hidden_size)
        self.hidden_to_update = nn.Linear(hidden_size, hidden_size)

        self.input_to_output = nn.Linear(input_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, hidden_size)

        self.input_to_cell_input = nn.Linear(input_size, hidden_size)
        self.hidden_to_cell_input = nn.Linear(hidden_size, hidden_size)

        # instantiate activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward_pass(self, x_input, hidden, cell_state):
        """
        This function implements the forward pass of a LSTM layer.

        :param x_input: The input to be evaluated
        :param hidden: Current hidden state
        :param cell_state: Current cell state
        :return: tuple of (output, hidden, cell_state) updated according to the LSTM equations
        """
        forget = self.sigmoid(self.input_to_forget(x_input) + self.hidden_to_forget(hidden))
        update = self.sigmoid(self.input_to_update(x_input) + self.hidden_to_update(hidden))
        output = self.sigmoid(self.input_to_output(x_input) + self.hidden_to_output(hidden))
        cell_input = self.tanh(self.input_to_cell_input(x_input) + self.hidden_to_cell_input(hidden))

        cell_state = forget * cell_state + update * cell_input
        hidden = output * self.tanh(cell_state)
        return output, hidden, cell_state


    def init_hidden_and_cell_state(self):
        """
        This function initialises the hidden and cell states as zeros.

        :return: tuple of (hidden, cell_state) consisting of zero vectors
        """
        return torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size)

class Model(nn.Module):
    def __init__(self, final_output_size, input_size, hidden_lstm_size, hidden_size):
        #TODO: Finish implementation
        super(Model, self).__init__()

# TODO: Implement training routine
# TODO: Export model
