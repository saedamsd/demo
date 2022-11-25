import re
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from scipy.sparse import csr_matrix
import hdbscan
from collections import defaultdict
from collections import Counter
from GMM_class import GMMClass

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def clean_text(
    sentence,
):
    pattern = re.compile(r'[^a-z]+')
    sentence = sentence.lower()
    sentence = pattern.sub(
        repl=' ',
        string=sentence,
    ).strip()
    word_list = word_tokenize(
        text=sentence,
    )
    stopwords_list = set(stopwords.words('english'))
    word_list = [word for word in word_list if word not in stopwords_list]
    word_list = [word for word in word_list if len(word) > 2]
    lemma = WordNetLemmatizer()
    word_list = [lemma.lemmatize(word) for word in word_list]
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
    return np.array(x_arr), np.array(y_arr)


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
    word2topics = pd.Series()
    for idx, component in enumerate(components):
        term_topic = pd.Series(
            component,
            index=terms,
        )
        term_topic.sort_values(
            ascending=False,
            inplace=True,
        )
        word2topics['topic ' + str(idx)] = [list(term_topic.iloc[:10].index), [v for v in term_topic.iloc[:10]]]
    return word2topics


def main(
    path='GMM_outputs.txt',
) -> int:
    dataset = fetch_20newsgroups(
        shuffle=True,
        random_state=32,
        remove=('headers', 'footers', 'qutes'),
    )
    news_df = pd.DataFrame(
        data={'News': dataset.data},
    )
    news_df['News'] = news_df['News'].apply(lambda x: clean_text(str(x)))
    news_df = news_df.reset_index()
    texts = [val['News'] for idx, val in news_df.iterrows()]
    max_ngram = 1
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
    lda_model = LatentDirichletAllocation(
        n_components=20,
        random_state=12,
        learning_method='online',
        max_iter=5,
        learning_offset=50,
    )
    lda_model.fit(matrix)
    doc_topic_lda = lda_model.transform(matrix)
    number_of_examples_per_cluster = 1e16
    rng = np.random.default_rng()
    clustering_algo = hdbscan.HDBSCAN(
        algorithm='best',
        alpha=1.0,
        approx_min_span_tree=True,
        gen_min_span_tree=True,
        metric='euclidean',
        min_cluster_size=30,
        min_samples=10,
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
    x_arr, y_arr = build_docs_cluster_labels(
        docs_per_cluster,
        doc_topic_lda,
    )
    gmm_inst = GMMClass(
        path='storage/',
        dataset_x=x_arr,
        dataset_y=y_arr,
    )
    gmm_inst.split_train_test_data(
        test_percent=0.25,
        random_state=42,
    )
    gmm_inst.GMM_init(
        colors=["navy", "turquoise"],
        n_clusters=21,
        random_state=0,
    )
    gmm_inst.GMM_train()
    gmm_inst.GMM_test_plot(
        fileplot='GMM_plot_fig_test_test',
    )
    return 0


if __name__ == '__main__':
    sys.exit(main('GMM_outputs.txt'))
