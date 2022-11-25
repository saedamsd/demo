import numpy as np
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
        dataset_x=[],
        dataset_y=[],
    ):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.n_classes = 0
        self.n_estimators = 0
        self.colors = []
        self.estimators = {}
        self.X_train = dataset_x
        self.X_test = []
        self.y_train = dataset_y
        self.y_test = []
        self.path = path
        self.fileop = fileop

    # Given a text and path and file name the function write the text to the file
    def save_gmm(
        self,
        text: str = '',
        path: str = None,
        fileop: str = None
    ):
        if path is not None:
            ver_path = path
        else:
            ver_path = self.path
        if fileop is not None:
            ver_filepath = fileop
        else:
            ver_filepath = self.fileop
        with open(
            file=ver_path+ver_filepath,
            mode='a',
        ) as f:
            f.write(
                text + '\n',
            )

    # Given the dataset of input features and labels, the function will split the data into a test and train set
    def split_train_test_data(
        self,
        test_percent,
        random_state=None,
    ):
        if random_state is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.dataset_x,
                    self.dataset_y,
                    test_size=test_percent,
                    random_state=random_state,
                )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(
                    self.dataset_x,
                    self.dataset_y,
                    test_size=test_percent,
                )
        self.X_train = self.X_train[self.y_train != -1]
        self.y_train = self.y_train[self.y_train != -1]

    # Given the gaussian mixture models and the colors, it will draw the ellipses on the plot
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
                covariances = np.diag(
                    v=gmm.covariances_[n][:2],
                )
            elif gmm.covariance_type == "spherical":
                covariances = np.eye(
                    N=gmm.means_.shape[1],
                ) * gmm.covariances_[n]
            else:
                return
            v, w = np.linalg.eigh(
                a=covariances,
            )
            u = w[0] / np.linalg.norm(
                x=w[0],
            )
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = mpl.patches.Ellipse(
                xy=gmm.means_[n, :2],
                width=v[0],
                height=v[1],
                angle=180 + angle,
                color=color,
            )
            ell.set_clip_box(
                clipbox=ax.bbox,
            )
            ell.set_alpha(
                alpha=0.5,
            )
            ax.add_artist(
                a=ell,
            )
            ax.set_aspect(
                aspect="equal",
                adjustable="datalim",
            )
        return ax

    # Given the number of clusters and the colors to be used this function will
    # init the parameters of the Gaussian Mixture models
    def GMM_init(
        self,
        colors,
        n_clusters,
        random_state=None,
    ):
        if random_state is not None:
            estimators = {
                cov_type: GaussianMixture(
                    n_components=n_clusters,
                    covariance_type=cov_type,
                    max_iter=20,
                    random_state=random_state,
                )
                for cov_type in ["spherical", "diag"]}
        else:
            estimators = {
                cov_type: GaussianMixture(
                    n_components=n_clusters,
                    covariance_type=cov_type,
                    max_iter=20,
                )
                for cov_type in ["spherical", "diag"]}
        self.n_estimators = len(estimators)
        self.colors = colors
        self.estimators = estimators
        return estimators

    # This function trains the inited GMM models based on the train set that
    # we saved in earlier calls of the split function
    def GMM_train(
        self,
    ):
        self.n_classes = len(np.unique(
            ar=self.y_train,
        ))
        for index, (name, estimator) in enumerate(self.estimators.items()):
            estimator.means_init = np.array(
                [self.X_train[self.y_train == i].mean(axis=0) for i in range(0, self.n_classes)],
            )
            estimator.fit(self.X_train)

    # Given the trained GMMs this function draws the  ellipses and the data points
    # and evaluate the trained model on the test set and report accuracy
    def GMM_test_plot(
        self,
        fileplot='GMM_plot_fig_test',
        test_x=None,
        test_y=None,
    ):
        if test_x is not None:
            self.X_test = test_x
        if test_y is not None:
            self.y_test = test_y
        n_estimators = len(self.estimators)
        for index, (name, estimator) in enumerate(self.estimators.items()):
            h = plt.subplot(
                2,
                n_estimators // 2,
                index + 1,
            )
            self.make_ellipses(
                gmm=estimator,
                ax=h,
                colors=self.colors,
            )
            for n, color in enumerate(self.colors):
                data = self.X_test[self.y_test == n]
                plt.scatter(
                    x=data[:, 0],
                    y=data[:, 1],
                    marker="x",
                    color=color,
                )
            y_test_pred = estimator.predict(
                self.X_test,
            )
            test_accuracy = np.mean(
                a=(y_test_pred.ravel() == self.y_test.ravel()) * 100)
            self.save_gmm("Test accuracy: %.1f" % test_accuracy)
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
            self.save_gmm("confusion matrix:\n")
            for i in range(cm.shape[0]):
                self.save_gmm(
                    text=str([cm[i, j] for j in range(cm.shape[1])]) + "\n",
                )
            plt.xticks(())
            plt.yticks(())
            plt.title(
                label=name,
            )
            plt.savefig(
                self.path+fileplot,
            )
            return test_accuracy
