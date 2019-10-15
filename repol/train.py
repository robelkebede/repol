from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer



def preprocesses():
    data = np.load("fana_data_700_new.npy")
    fea = []
    features = data[:,0]
    target = data[:,1:9]


    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    y_data = []
    stop_words = set(stopwords.words("english"))
    count_vectorizer = CountVectorizer()


    for sentence in features:
        text = nltk.sent_tokenize(sentence)
        bag_of_words = count_vectorizer.fit_transform(text)

        """
        LETER
        for t in text:
            t=stemmer.stem(t)
            new_text = lemmatizer.lemmatize(t, wordnet.VERB)
        """

        feature_names = count_vectorizer.get_feature_names()

        fea.append(bag_of_words)

    fea = np.array(fea)
    print(fea.shape)
    print(target.shape)

    return fea,target


    

def train():


    x,y = preprocesses()

    X2 = np.random.random((7,3))

    svr = SVR(gamma="auto")
    regr = MultiOutputRegressor(svr)

    regr.fit(x,y)

def main():
    train()

if __name__ == "__main__":
    main()
