from unittest import TestCase

import torch

from train_model import sentence_to_tensor


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


