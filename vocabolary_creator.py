import re
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# import umap
from sklearn.decomposition import TruncatedSVD, PCA, NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# !pip install gensim
# from gensim import corpora
# from gensim.models.ldamodel import LdaModel

import numpy as np
import pandas as pd

import nltk
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class create_vocab_class:
    def __init__(
            self,
            vocabulary,
    ):
        self.dataset = {}
        self.vocabulary = {}

    def clean_and_tokenize_text(self, sentence, ):
        # remove non alphabetic sequences
        pattern = re.compile(r'[^a-z]+')
        sentence = sentence.lower()
        sentence = pattern.sub(' ', sentence).strip()

        # Tokenize
        word_list = word_tokenize(sentence)
        return word_list

    def remove_stopwords_and_shorts_text(self, word_list, ):

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
        return word_list

    def lemmatize_text(self, word_list, ):

        # lemmatize
        lemma = WordNetLemmatizer()
        word_list = [lemma.lemmatize(word) for word in word_list]
        # list to sentence
        sentence = ' '.join(word_list)

        return sentence

    def preprocess_text(self, inp_sentence, ):

        word_list = self.clean_and_tokenize_text(inp_sentence)
        word_list =  self.remove_stopwords_and_shorts_text(word_list)
        sentence = self. lemmatize_text(word_list)
        return sentence

    def import_data_set(self, dataset_name):

        if dataset_name == '20newsgroup':
            # Load news data set
            # remove meta data headers footers and quotes from news dataset

            dataset = fetch_20newsgroups(shuffle=True,
                                        random_state=32,
                                        remove=('headers', 'footers', 'qutes'))
        else:
            dataset = {}
        # put your data into a dataframe
        news_df = pd.DataFrame({'News': dataset.data
                               })

        self.dataset = dataset
        return news_df

    def create_df(self, news_df, ):


        # preprocess text data
        news_df['News'] = news_df['News'].apply(lambda x: self.preprocess_text(str(x)))

        words = [item for idx,val in news_df.iterrows() for item in val['News'].split()]

        return words

    def create_vocab(self, words, ):

        news_df = self.import_data_set('20newsgroup')
        words = self.create_df(news_df)

        vocabulary = dict(
            zip(
                sorted(words),
                np.arange(len(words)),
            ),
        )
        self.vocabulary = vocabulary
        return vocabulary


    def print_dict(self,
            path: str = '/vocab.txt',
        ):
            with open(path, 'a') as f:
                for word, cnt in self.vocabulary:
                    f.write(word + ' ' + cnt +'\n')
