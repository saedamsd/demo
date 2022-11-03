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
    ) -> None:
        self.dataset_name = '20newsgroup'

    def tearDown(
            self,
    ):
        pass

    def test_import_data_set(self, dataset_name):

        with mock.patch(
                target='sklearn.datasets.fetch_20newsgroups',
        ) as fetch_20newsgroups:

            fetch_20newsgroups.return_value.data = ['first sentence','second sentence']

        vocab_inst = create_vocab_class()
        vocab_inst_df = vocab_inst.import_data_set(dataset_name)

        static_data = pd.DataFrame({'News': ['first sentence','second sentence']
                      })

        self.assertEqual(
            first=vocab_inst_df,
            second=static_data,
        )


    #Saeed - the test_ should be there in evenry function name so the test can run !!!!
    def test_create_vocab(self):
        pass


    def test_print_vocab(self):
        pass

if __name__ == '__main__':
    unittest.main()