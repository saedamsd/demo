import re
import string
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA, NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from scipy.sparse import csr_matrix
import os
import hdbscan
from collections import defaultdict
from collections import Counter
from time import time
from random import random
from sklearn.metrics.pairwise import cosine_distances

from GMM_class import GMMClass

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def clean_text(
        sentence,
):
    # remove non alphabetic sequences
    pattern = re.compile(r'[^a-z]+')
    sentence = sentence.lower()
    sentence = pattern.sub(
        ' ',
        sentence
    ).strip()

    # Tokenize
    word_list = word_tokenize(sentence)

    # stop words
    stopwords_list = set(stopwords.words('english'))
    # puctuation
    # punct = set(string.punctuation)

    # remove stop words
    word_list = [word for word in word_list if word not in stopwords_list]
    # remove very small words, length < 3
    # they don't contribute any useful information
    word_list = [word for word in word_list if len(word) > 2]
    # remove punctuation
    # word_list = [word for word in word_list if word not in punct]

    # stemming
    # ps  = PorterStemmer()
    # word_list = [ps.stem(word) for word in word_list]

    # lemmatize
    lemma = WordNetLemmatizer()
    word_list = [lemma.lemmatize(word) for word in word_list]
    # list to sentence
    sentence = ' '.join(word_list)

    return sentence


def build_docs_cluster_labels(
        docs_per_cluster,
        doc_topic_lda,
):
    x_arr = list()
    y_arr = list()
    for cluster_lbl in docs_per_cluster:
        for doc in docs_per_cluster[cluster_lbl]:
            x_arr.append(list(doc_topic_lda[doc]))
            y_arr.append(cluster_lbl)
    return np.array(x_arr),np.array(y_arr)


def preprocessor(
            text: str,
    ) -> str:
        text = text.lower()

        text = re.sub(r'(?<=[\s])[:\-&\(\)+\'\*_<=>|•`?«!\]\[“\",\.;~]+', '', text)
        text = re.sub(r'[:\-&\(\)+\'*_<=>|•`?«!\]\[“\",.;~]+(?=[\s])', '', text)
        text = re.sub(r'(?<=\A\b)[:\-&\(\)+\'\*_<=>|•`?«!\]\[“\",\.;~]+', '', text)
        text = re.sub(r'[:\-&\(\)+\'*_<=>|•`?«!\]\[“\",.;~]+(?=\Z)', '', text)

        text = re.sub(r'(?<=[\s])[0-9]+(\,[0-9]{3})*[\.][0-9]+(?=[\s])', 'FLOAT', text)
        text = re.sub(r'(?<=[\s])[0-9]+(\,[0-9]{2}){1}(?=[\s])', 'FLOAT', text)
        text = re.sub(r'(?<=[\s])[0-9]+(\,[0-9]{3})*(?=[\s])', 'INT', text)

        text = re.sub(r'(?<=\A\b)[0-9]+(\,[0-9]{3})*[\.][0-9]+(?=\Z\b)', 'FLOAT', text)
        text = re.sub(r'(?<=\A\b)[0-9]+(\,[0-9]{2}){1}(?=\Z\b)', 'FLOAT', text)
        text = re.sub(r'(?<=\A\b)[0-9]+(\,[0-9]{3})*(?=\Z\b)', 'INT', text)

        return text


# function to map words to topics
def map_word2topic(
        components,
        terms,
):
    # create output series
    word2topics = pd.Series()
    word2topics_proba = pd.Series()

    for idx, component in enumerate(components):
        # map terms (words) with topic
        # which is probability of word given a topic P(w|t)
        term_topic = pd.Series(
            component,
            index=terms,
        )
        print(term_topic)
        # sort values based on probability
        term_topic.sort_values(
            ascending=False,
            inplace=True,
        )
        print(term_topic)

        # put result in series output
        word2topics['topic ' + str(idx)] = [list(term_topic.iloc[:10].index), [v for v in term_topic.iloc[:10]]]

        print(term_topic.iloc[:10].index)
        print([v for v in term_topic.iloc[:10]])

        print(word2topics)

    return word2topics


class ClusterCoordinateGenerator:
    def __init__(
            self,
    ):
        self.coordinates = []
        self.max_attempts = 40
        self.min_cluster_distance = 0.2

    def generate_coordinates(
            self,
            num_of_clusters,
    ):
        list_coordinates = [
            self.get_next_cluster_position()
            for _ in range(num_of_clusters)
        ]

        return list_coordinates

    def get_random_coordinate(
            self,
    ):
        use_positive = random() < 0.5

        if use_positive:
            return random() * 0.8

        return random() * 0.8 * -1

    def is_far_enough(
            self,
            coordinates,
            point,
    ):
        for coordinate in coordinates:
            #             if dist(coordinate, point) < self.min_cluster_distance:
            if cosine_distances(coordinate, point) < self.min_cluster_distance:
                return False

        return True

    def get_next_cluster_position(
            self,
    ):
        attempts = 0
        random_x = self.get_random_coordinate()
        random_y = self.get_random_coordinate()

        is_finished_generating_clusters = False
        while not is_finished_generating_clusters:
            random_x = self.get_random_coordinate()
            random_y = self.get_random_coordinate()
            attempts += 1

            is_far_enough = self.is_far_enough(
                coordinates=self.coordinates,
                point=[
                    random_x,
                    random_y,
                ],
            )

            if is_far_enough:
                is_finished_generating_clusters = True

            if attempts > self.max_attempts:
                is_finished_generating_clusters = True

        new_coordinate = [
            random_x,
            random_y,
        ]
        self.coordinates.append(new_coordinate)

        coordinates = {
            'x': random_x,
            'y': random_y,
        }

        return coordinates


def main(
       path='GMM_outputs.txt',
) -> int:


    # Load news data set
    # remove meta data headers footers and quotes from news dataset
    dataset = fetch_20newsgroups(
        shuffle=True,
                                 random_state=32,
                                 remove=('headers', 'footers', 'qutes'),
    )

    # put your data into a dataframe
    news_df = pd.DataFrame({'News': dataset.data})
    news_df['News'] = news_df['News'].apply(lambda x: clean_text(str(x)))

    news_df = news_df.reset_index()
    processed_documents = 0

    texts = [val['News'] for idx, val in news_df.iterrows()]

    max_ngram = 1

    # stop_words = frozenset
    stop_words = {'english'}

    vectorizer = CountVectorizer(
        ngram_range=(1, max_ngram),
        stop_words=stop_words,
        preprocessor=preprocessor,
        tokenizer=nltk.word_tokenize,
        token_pattern=None,
        max_df=0.95,
        min_df=0.005,
    )

    vectorizer.fit_transform(texts)

    engine = vectorizer

    vector = engine.transform(
        raw_documents=texts,
    )

    result = csr_matrix(vector)
    result.eliminate_zeros()

    matrix = result

    # lda instance
    lda_model = LatentDirichletAllocation(
        n_components=20,  # number of topics
        random_state=12,
        learning_method='online',
        max_iter=5,  # maximum number of passes epochs on the dataset
        learning_offset=50,
    )

    lda_model.fit(matrix)

    doc_topic_lda = lda_model.transform(matrix)

    word2topics_lda = map_word2topic(
        lda_model.components_,
        vectorizer.get_feature_names(),
    )

    n_clusters = 30
    number_of_examples_per_cluster = 1e16

    t0 = time()
    rng = np.random.default_rng()

    clustering_algo = hdbscan.HDBSCAN(
        algorithm='best', alpha=1.0, approx_min_span_tree=True,
        gen_min_span_tree=True, metric='euclidean', min_cluster_size=30, min_samples=10,
        cluster_selection_method='eom',
    )

    topic_matrix = doc_topic_lda

    file_to_index_dict = {}

    for i in range(len(doc_topic_lda)):
        file_to_index_dict[i] = i


    labels = clustering_algo.fit_predict(
        X=topic_matrix,
    )

    counts = Counter(labels)

    docs_per_cluster = defaultdict(list)

    for running_index, (path, index) in enumerate(file_to_index_dict.items()):
        label = int(labels[running_index])
        docs_per_cluster[label].append(path)

    graph = {}

    for cluster_index in counts.keys():
        if cluster_index == -1:
            continue
        output_cluster_num = cluster_index + 1
        cluster_radius = counts[cluster_index]
        cluster_list_size = np.minimum(
            len(docs_per_cluster[cluster_index]),
            number_of_examples_per_cluster,
        )
        cluster_file_list = rng.choice(
            a=docs_per_cluster[cluster_index],
            size=int(cluster_list_size),
            replace=False,
        )
        graph[output_cluster_num.item()] = {
            'r': int(cluster_radius),
            'files_list': list(cluster_file_list),
        }

    new_clusters = graph

    number_of_clusters = len(new_clusters)

    x_arr, y_arr = build_docs_cluster_labels(docs_per_cluster, doc_topic_lda)

    gmm_inst = GMMClass('GMM_outputs.txt')
    gmm_inst.split_train_test_data(x_arr, y_arr)
    gmm_inst.GMM_init(["navy", "turquoise"], 2)
    gmm_inst.GMM_train()
    gmm_inst.GMM_test_plot()

    return 0


if __name__ == '__main__':
    sys.exit(main('GMM_outputs.txt'))  # next section explains the use of sys.exit
