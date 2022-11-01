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
        self.vocabulary = {}

    def clean_text(sentence):
        # remove non alphabetic sequences
        pattern = re.compile(r'[^a-z]+')
        sentence = sentence.lower()
        sentence = pattern.sub(' ', sentence).strip()

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
    def create_vocab(self, ):
        # Load news data set
        # remove meta data headers footers and quotes from news dataset
        dataset = fetch_20newsgroups(shuffle=True,
                                    random_state=32,
                                    remove=('headers', 'footers', 'qutes'))

            # clean text data
            # remove non alphabetic characters
            # remove stopwords and lemmatize

        # put your data into a dataframe
        news_df = pd.DataFrame({'News': dataset.data
        #                         ,
        #                        'Target': dataset.target
                               })

        # get dimensions of data
        news_df.shape

        news_df['News'] = news_df['News'].apply(lambda x: self.clean_text(str(x)))

        # we'll use tqdm to monitor progress of data cleaning process
        # create tqdm for pandas
        tqdm.pandas()
        # clean text data
        news_df['News'] = news_df['News'].apply(lambda x: self.clean_text(str(x)))

        words = [item for idx,val in news_df.iterrows() for item in val['News'].split()]

        self.vocabulary = dict(
            zip(
                sorted(words),
                np.arange(len(words)),
            ),
        )


    def print_dict(self,
            path: str = '/vocab.txt',

        ):
            with open(path, 'a') as f:
                for word, cnt in self.vocabulary:
                    f.write(word + ' ' + cnt +'\n')
