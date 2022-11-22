import json
import logging
import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix


class GMMClass:
    def __init__(
            self,
            path='storage/',
            fileop: str = 'GMM_outputs.txt',
            fileplot: str = 'GMM_plot_fig_test',
    ):
        self.n_classes = 0
        self.n_estimators = 0
        self.colors = []
        self.estimators = {}
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.path = path
        self.fileop = fileop
        isExist = os.path.exists(self.path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path)

    def save_GMM(
            self,
            text: str = '',
            path: str = None,
            fileop: str = None
    ):
        # self.path = path
        if path != None:
            Path = path
        else:
            Path = self.path
        if fileop != None:
            filePath = fileop
        else:
            filePath = self.fileop
        with open(
                self.path+filePath,
                'a',
        ) as f:
            f.write(text + '\n')

    def split_train_test_data(
            self,
            topic_matrix,
            labels,
    ):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(
                topic_matrix,
                labels,
                test_size=0.25,
                random_state=42,
            )
        self.X_train = self.X_train[self.y_train != -1]
        self.y_train = self.y_train[self.y_train != -1]

    def make_ellipses(
            self,
            gmm,
            ax,
            colors,
    ):
        for n, color in enumerate(colors):
            if gmm.covariance_type == "full":
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == "tied":
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == "diag":
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == "spherical":
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            else:
                return
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = mpl.patches.Ellipse(
                gmm.means_[n, :2],
                v[0],
                v[1],
                180 + angle,
                color=color,
            )
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            ax.set_aspect("equal",
                          "datalim",
                          )

        return ax

    def GMM_init(
            self,
            colors,
            n_clusters,
    ):
        estimators = {
            cov_type: GaussianMixture(
                n_components=n_clusters,
                covariance_type=cov_type,
                max_iter=20,
                random_state=0,
            )
            for cov_type in ["spherical", "diag"]}

        self.n_estimators = len(estimators)
        self.colors = colors
        self.estimators = estimators
        return estimators


    def GMM_train(
            self,
    ):

        self.n_classes = len(np.unique(self.y_train))

        for index, (name, estimator) in enumerate(self.estimators.items()):
            estimator.means_init = np.array(
                [self.X_train[self.y_train == i].mean(axis=0) for i in range(0, self.n_classes)]
            )
            estimator.fit(self.X_train)

    def GMM_test_plot(
            self,
            fileplot = 'GMM_plot_fig_test',
    ):
        n_estimators = len(self.estimators)
        for index, (name, estimator) in enumerate(self.estimators.items()):
            h = plt.subplot(2, n_estimators // 2, index + 1)
            self.make_ellipses(estimator, h, self.colors)
            for n, color in enumerate(self.colors):
                data = self.X_test[self.y_test == n]
                plt.scatter(
                    data[:, 0],
                    data[:, 1],
                    marker="x",
                    color=color,
                )

            y_test_pred = estimator.predict(self.X_test)
            test_accuracy = np.mean(y_test_pred.ravel() == self.y_test.ravel()) * 100
            self.save_GMM("Test accuracy: %.1f" % test_accuracy)
            plt.text(
                0.05,
                0.8,
                "Test accuracy: %.1f" % test_accuracy,
                transform=h.transAxes,
            )
            cm = confusion_matrix(
                self.y_test,
                y_test_pred,
            )
            self.save_GMM("confusion matrix:\n" % cm)
            plt.xticks(())
            plt.yticks(())
            plt.title(name)
            plt.savefig(self.path+fileplot)
            return test_accuracy

