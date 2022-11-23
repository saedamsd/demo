import unittest
import unittest.mock as mock

import matplotlib.axes
import numpy as np
import os
import GMM_class
from GMM_class import GMMClass
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class TestGMMClass(unittest.TestCase):

    def setUp(
            self,
    ) -> None:
        self.dataset_name = '20newsgroup'
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.topic_matrix = []
        self.labels = []
        self.colors = ["navy", "turquoise"]

        self.topic_matrix = np.array(
            [[0.00166667, 0.00166667, 0.00166667, 0.14958999, 0.00166667, 0.00166667],
             [0.00166667, 0.00166667, 0.00166667, 0.14958999, 0.00166667, 0.00166667],
             [0.00166667, 0.00166667, 0.00166667, 0.14958999, 0.00166667, 0.00166667],
             [0.00166667, 0.00166667, 0.00166667, 0.14958999, 0.00166667, 0.00166667], ]
        )

        self.labels = np.array([1, 0, 1, 0])
        try:
            os.remove("storage/GMM_outputs_test.txt")
            os.remove("storage/GMM_plot_fig_test_test.png")
        except FileNotFoundError:
            pass

    def tearDown(self):
        try:
            os.remove("storage/GMM_outputs_test.txt")
            os.remove("storage/GMM_plot_fig_test_test.png")
        except FileNotFoundError:
            pass

    def test_save_GMM(
            self,
            path: str = 'storage/',
            fileop: str = 'GMM_outputs_test.txt',
    ):
        gmm_instance = GMMClass(
            path=path,
            fileop=fileop,
            dataset_x=self.topic_matrix,
            dataset_y=self.labels,
        )

        gmm_instance.save_GMM(
            'text for printing',
            path,
            fileop,
        )
        isFile = os.path.isfile(path+fileop)
        file = open(
            path+fileop,
            'r',
        )
        self.assertEquals(
            isFile,
            True,
        )
        self.assertEquals(
            file.read(),
            'text for printing\n',
        )
        file.close()

    def test_split_train_test_data(
            self,
            test_percent=0.25,
    ):
        gmm_instance = GMMClass(
            dataset_x=self.topic_matrix,
            dataset_y=self.labels,
        )
        gmm_instance.split_train_test_data(
            test_percent,
        )

        self.assertIsNotNone(gmm_instance.X_train)
        self.assertIsNotNone(gmm_instance.y_train)
        self.assertIsNotNone(gmm_instance.X_test)
        self.assertIsNotNone(gmm_instance.y_test)

        self.assertIs(
            type(gmm_instance.X_train),
            np.ndarray,
        )
        self.assertIs(
            type(gmm_instance.y_train),
            np.ndarray,
        )
        self.assertIs(
            type(gmm_instance.X_test),
            np.ndarray,
        )
        self.assertIs(
            type(gmm_instance.y_test),
            np.ndarray,
        )
        x_len = len(self.topic_matrix)
        y_len = len(self.labels)
        x_test_len = len(gmm_instance.X_test)
        y_test_len = len(gmm_instance.y_test)
        self.assertEquals(
            x_test_len / x_len,
            test_percent,
        )
        self.assertEquals(
            y_test_len / y_len,
            test_percent,
        )

    def test_make_ellipses(
            self,
    ):
        with mock.MagicMock(
                target='sklearn.mixture.GaussianMixture', ) as gmm_magic_mock:
            gmm_instance = GMMClass(
                dataset_x=self.topic_matrix,
                dataset_y=self.labels,
            )

            h = plt.subplot(
                2,
                2 // 2,
                1 + 1,
            )
            try:
                gmm_instance.make_ellipses(
                    gmm_magic_mock,
                    h,
                    ["navy", "turquoise"],
            )
                self.assertTrue(True)
            except:
                self.assertTrue(False)

    def test_GMM_init(
            self,
    ):
        gmm_instance = GMMClass(
            dataset_x=self.topic_matrix,
            dataset_y=self.labels,
        )

        gmm_instance.GMM_init(
            ["navy", "turquoise"],
            2,
        )
        self.assertEqual(
            ["navy", "turquoise"],
            gmm_instance.colors,
        )
        self.assertNotEqual(
            None,
            gmm_instance.estimators,
        )

    def test_GMM_train(
            self,
    ):

        with mock.patch(
                target='GMM_class.GMMClass', ) as GMMClass_mock:
            GMMClass_mock.return_value.GMM_validation_plot.return_value = None
            gmm_instance = GMMClass(
                dataset_x=self.topic_matrix,
                dataset_y=self.labels,
            )
            try:
                gmm_instance.GMM_train()
                self.assertTrue(True)
            except:
                self.assertTrue(False)


    def test_GMM_test_plot(
            self,
    ):
        with mock.patch(
                target='GMM_class.GMMClass', ) as GMMClass_mock, mock.patch(
                target='GMM_class.np', ) as np_mock, mock.patch(
                target='GMM_class.GMMClass.GMM_validation_plot.len', ) as len_mock, mock.patch(
                target='GMM_class.GaussianMixture', ) as GaussianMixture_mock:

            GMMClass_mock.return_value.make_ellipses.return_value = None
            GaussianMixture_mock.covariance_type = None
            gmm_instance = GMMClass(
                path="storage/",
                fileop="GMM_outputs_test.txt",
                dataset_x=self.topic_matrix,
                dataset_y=self.labels,
            )

            gmm_instance.X_test = self.topic_matrix
            gmm_instance.y_test = self.labels
            GaussianMixture_mock.return_value.predict.return_value = gmm_instance.y_test
            gmm_instance.GMM_init(
                colors=['red'],
                n_clusters=1)

            gmm_instance.GMM_test_plot('GMM_plot_fig_test_test')
            path = "storage/GMM_plot_fig_test_test.png"
            isFile = os.path.isfile(path)
            self.assertEquals(isFile, True)
            isGmmFile = os.path.isfile("storage/GMM_outputs_test.txt")
            self.assertEquals(isGmmFile, True)


if __name__ == '__main__':
    unittest.main()
