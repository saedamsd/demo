import unittest
import unittest.mock as mock

from unittest.mock import patch
import pandas as pd

from vocabolary_creator import create_vocab_class


class TestVocabClass(unittest.TestCase):

    #For the class not the instance
    def setUpClass(
            cls,
    ):

        pass
    def tearDownClass(
            cls,
    ):
        pass

    def setUp(
            self,
    ):
        self.vocab = {}
    def tearDown(
            self,
    ):
        pass


    #Saeed - the test_ should be there in evenry function name so the test can run !!!!
    def test_create_vocab(self):

        # with mock.patch(
        #         target='demo.create_vocab_class.create_vocab_class.vocabulary',
        # ) as vocab

        self.vocab = create_vocab_class()
        self.vocab = self.vocab.create_vocab('20newsgroup')

    def test_print_vocab(self):
        self.vocab.print_dict('output')


if __name__ == '__main__':
    unittest.main()