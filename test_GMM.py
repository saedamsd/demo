import unittest
import unittest.mock as mock

from unittest.mock import patch
import pandas as pd

from vocabolary_creator import create_vocab_class

from GMM_class import GMM_class_


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

    def test_save_GMM(
        self,
        path: str = '/storage/GMM_outputs.txt',
        text: str = ''

    ):
        pass

    def test_split_train_test_data(self, topic_matrix, labels, ):
        pass

    def test_make_ellipses(self, gmm, ax, colors, ):
        pass

    def test_GMM_init(self, colors, n_clusters ):
        with mock.patch(
                target='sklearn.mixture.GaussianMixture',
        ) as GaussianMixture_func:
            GaussianMixture_func.return_value = None

        gmm_instance = GMM_class_()
        estimators = gmm_instance.GMM_init(self, ["navy", "turquoise"], 2)

        self.assertEqual(
            first=estimators,
            second=None,
        )

    def test_GMM_validation_plot(self, X_train, y_train, estimator, name, index, h,):
        with mock.patch(
                target='matplotlib.pyplot.subplot',
        ) as subplot_mock, mock.patch(
                target='GMM_class.GMM_class_.make_ellipses',) as make_ellipses_mock:

            subplot_mock.return_value = None
            make_ellipses_mock.return_value = None

    def test_GMM_train(self, X_train, y_train, colors, ):
        pass

    def test_GMM_test_plot(self, X_test, y_test, colors, h, ):
        pass



if __name__ == '__main__':
    unittest.main()