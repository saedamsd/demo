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
        except FileNotFoundError:
            pass

    def tearDown(self):
        try:
            os.remove("storage/GMM_outputs_test.txt")
        except FileNotFoundError:
            pass

    def test_save_GMM(
            self,
            path: str = 'storage/',
            fileop: str = 'GMM_outputs_test.txt',
    ):
        gmm_instance = GMMClass(path=path,fileop=fileop)
        gmm_instance.save_GMM(
            'text for printing',
            path,
            fileop,
        )
        isFile = os.path.isfile(path+fileop)
        file = open(path+fileop, 'r')
        self.assertEquals(isFile, True)
        self.assertEquals(file.read(), 'text for printing\n')
        file.close()
        # if my_file.is_file():
        # assert file exists
        # assert if file contains values
        # delete file

    def test_split_train_test_data(
            self,
    ):
        gmm_instance = GMMClass()
        gmm_instance.split_train_test_data(
            self.topic_matrix,
            self.labels,
        )

        self.assertIsNotNone(gmm_instance.X_train)
        self.assertIsNotNone(gmm_instance.y_train)
        self.assertIsNotNone(gmm_instance.X_test)
        self.assertIsNotNone(gmm_instance.y_test)

        self.assertIs(type(gmm_instance.X_train), np.ndarray)
        self.assertIs(type(gmm_instance.y_train), np.ndarray)
        self.assertIs(type(gmm_instance.X_test), np.ndarray)
        self.assertIs(type(gmm_instance.y_test), np.ndarray)
        x_len = len(self.topic_matrix)
        y_len = len(self.labels)
        x_test_len = len(gmm_instance.X_test)
        y_test_len = len(gmm_instance.y_test)
        self.assertEquals(x_test_len / x_len, 0.25)
        self.assertEquals(y_test_len / y_len, 0.25)

        # asset t_train not -1
        # assert x_train not -1
        # check ratio of x_train and y_train to test xy allow for ratio to be +/- 0.05 to be sure
        # check the labels


    def test_make_ellipses(
            self,
    ):
        with mock.MagicMock(
                target='sklearn.mixture.GaussianMixture', ) as gmm_magic_mock:
            gmm_instance = GMMClass()
            # gmm_magic_mock.covariance_type = "spherical"
            h = plt.subplot(
                2,
                2 // 2,
                1 + 1,
            )

            mpl_ax = gmm_instance.make_ellipses(
                gmm_magic_mock,
                h,
                ["navy", "turquoise"],
            )
            self.assertIs(mpl_ax, None)
            # self.assertIs(mpl_ax, matplotlib.axes.Axes)
            # assert what it returns


    def test_GMM_init(
            self,
    ):
        gmm_instance = GMMClass()
        gmm_instance.GMM_init(
            ["navy", "turquoise"],
            2,
        )
        self.assertEqual(["navy", "turquoise"], gmm_instance.colors)
        self.assertNotEqual(None, gmm_instance.estimators)

    def test_GMM_train(
            self,
    ):

        with mock.patch(
                target='GMM_class.GMMClass', ) as GMMClass_mock:
            GMMClass_mock.return_value.GMM_validation_plot.return_value = None
            # why this mock is needed GMM_train does not use validation plot internatlly , so whats the point of this ?
            gmm_instance = GMMClass()
            try:
                gmm_instance.GMM_train()
                self.assertTrue(True)
            except:
                self.assertTrue(False)

            # this does not return anything , for testing to be done ,
            # a functon must either return a value , or manipulate its own value which can be checked

    def test_GMM_test_plot(
            self,
    ):
        with mock.patch(
            target='GMM_class.plt',
        ) as pyplot_mock, mock.patch(
                target='GMM_class.GMMClass', ) as GMMClass_mock, mock.patch(
                target='GMM_class.np', ) as np_mock, mock.patch(
                target='GMM_class.GMMClass.GMM_validation_plot.len', ) as len_mock, mock.patch(
                # target='GMM_class.GaussianMixture.predict',spec=np_mock.array([1, 0, 1, 0]) ) as GaussianMixture_predict_mock,mock.patch(
                target='GMM_class.GaussianMixture', ) as GaussianMixture_mock,mock.patch(
                target='sklearn.metrics.confusion_matrix',return_value = 'Mocked This Silly' ) :
            np_mock.side_effect =[1, 0, 1, 0]
            np_mock.spec = [1, 0, 1, 0]
            np_mock.return_value = [1, 0, 1, 0]
        #     # pyplot_mock.return_value = None
        #     confusion_matrix_mock=metrics.return_value
        #     confusion_matrix_mock.\
        #     metrics_mock.confusion_matrix = None

            # confusion_matrix_mock.confusion_matrix.return_value = []
            GMMClass_mock.return_value.make_ellipses.return_value = None
            GaussianMixture_mock.covariance_type = None

        #     # GMMClass_mock.return_value.save_GMM_mock.return_value = None
        #     GMMClass_mock.return_value.confusion_matrix_mock.return_value = None
        #     GMMClass_mock.return_value.GMM_validation_plot.return_value = None
        #     GaussianMixture_predict_mock.return_value =  np_mock.array([1, 0, 1, 0])
        #     len_mock.return_value = 10
        #     len_mock.return_value = 10
            gmm_instance = GMMClass(path="storage/",fileop="GMM_outputs_test.txt")
            gmm_instance.split_train_test_data(
                self.topic_matrix,
                self.labels,
            )
            GaussianMixture_mock.return_value.predict.return_value = gmm_instance.y_test
            gmm_instance.GMM_init(
                colors=['red'],
                n_clusters=1)

            accuracy = gmm_instance.GMM_test_plot()
            path = "storage/GMM_plot_fig_test"
            isFile = os.path.isfile(path)
            # self.assertEquals(isFile, True)
            #print(gmm_instance.path+"GMM_outputs.txt")
            isGmmFile = os.path.isfile("storage/GMM_outputs_test.txt")
            self.assertEquals(isGmmFile, True)
            # /storage/GMM_plot_fig_test
            # confusion matrix in the file Save_GMM


if __name__ == '__main__':
    unittest.main()
