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

class RNN(nn.module):
    pass

# TODO: Implement training routine
# TODO: Export model
