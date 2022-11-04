import json
import logging
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

class GMM_class_:
    def __init__(
            self,
            vocabulary,
    ):
        self.n_classes = 0
        self.n_estimators = 0
        self.colors = ["navy", "turquoise"]
        self.estimators = None

    #  Saeed - see how to get the code of GMM here since now we have the labels from get_labels

    def save_GMM(
            self,
            path: str = '/storage/GMM_outputs.txt',
            text: str = ''
    ):
        with open(path, 'a') as f:
            f.write(text + '\n')

    # Saeed - commented one here as np.unique(labels) is 3
    # saeed opened again same error below - Error - IndexError: index 7 is out of bounds for axis 0 with size 7
    # , "green"
    #  ]

    def split_train_test_data(self, topic_matrix, labels, ):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(topic_matrix, labels, test_size=0.25,
                                                                                random_state=42)
        # Saeed - removing the -1 samples points only from the train data and commenting the above y_arr!=-1
        self.X_train = self.X_train[self.y_train != -1]
        self.y_train = self.y_train[self.y_train != -1]

    def make_ellipses(gmm, ax, colors):
        for n, color in enumerate(colors):
            if gmm.covariance_type == "full":
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == "tied":
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == "diag":
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == "spherical":
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = mpl.patches.Ellipse(
                gmm.means_[n, :2], v[0], v[1], 180 + angle, color=color
            )
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            ax.set_aspect("equal", "datalim")

    def GMM_init(self, colors, n_clusters, ):
        # Try GMMs using different types of covariances.
        estimators = {
            cov_type: GaussianMixture(n_components=n_clusters, covariance_type=cov_type, max_iter=20, random_state=0)
            # Saeed - here for showing only one to fix the accuracy issue
            # for cov_type in ["spherical", "diag", "tied", "full"]
            for cov_type in ["spherical", "diag"]}

        self.n_estimators = len(estimators)
        self.colors = colors
        self.estimators = estimators
        return estimators

    def GMM_validation_plot(self, X_train, y_train, estimator, name, index, h,):
            h = plt.subplot(2, self.n_estimators // 2, index + 1)
            self.make_ellipses(estimator, h, self.colors)
            for n, color in enumerate(self.colors):
                data = X_train[y_train == n]
                plt.scatter(
                    data[:, 0], data[:, 1], s=0.8, color=color, label=X_train[n]
                )
            y_train_pred = estimator.predict(X_train)
            train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
            plt.text(0.05, 0.9, "Train accuracy: %.1f" % train_accuracy, transform=h.transAxes)

            plt.xticks(())
            plt.yticks(())
            plt.title(name)
            self.save_GMM("Train accuracy: %.1f" % train_accuracy)
            plt.savefig('/storage/GMM_plot_fig_train')

    def GMM_train(self, X_train, y_train, colors, ):

        self.n_classes = len(np.unique(y_train))

        for index, (name, estimator) in enumerate(self.estimators.items()):
            # Since we have class labels for the training data, we can
            # initialize the GMM parameters in a supervised manner.

            estimator.means_init = np.array(
                # Saeed - changed ot run over -1 to 5 not as here beecause of error below on the fit():
                # [X_train[y_train == i].mean(axis=0) for i in range(-1,n_classes-1)]
                [X_train[y_train == i].mean(axis=0) for i in range(0, self.n_classes)]
            )
            # Train the other parameters using the EM algorithm.
            estimator.fit(X_train)

            self.GMM_validation_plot(self, X_train, y_train, estimator, name)

    def GMM_test_plot(self, X_test, y_test, colors, h, ):
        n_estimators = len(self.estimators)
        for index, (name, estimator) in enumerate(self.estimators.items()):

            h = plt.subplot(2, n_estimators // 2, index + 1)
            self.make_ellipses(estimator, h, colors)
            # Plot the test data with crosses
            for n, color in enumerate(colors):
                data = X_test[y_test == n]
                plt.scatter(data[:, 0], data[:, 1], marker="x", color=color)

            y_test_pred = estimator.predict(X_test)
            test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
            self.save_GMM("Test accuracy: %.1f" % test_accuracy)
            plt.text(0.05, 0.8, "Test accuracy: %.1f" % test_accuracy, transform=h.transAxes)

            cm = confusion_matrix(y_test, y_test_pred)
            self.save_GMM("confusion matrix:\n" % cm)

            plt.xticks(())
            plt.yticks(())
            plt.title(name)
            plt.savefig('/storage/GMM_plot_fig_test')

