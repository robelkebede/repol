from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn import metrics
import joblib
import string
import pickle



def preprocesses():
    #data = np.load("fana_data_700_new.npy")
    data = np.load("./data/700.npy",allow_pickle=True)
    new_data = []
    fea = data[0]
    target = data[1]

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    cv = CountVectorizer()

    stopWords = set(stopwords.words('english'))

    for i,f in enumerate(fea):
        sent = nltk.sent_tokenize(f) 

        for s in sent:
            panc = [char for char in s if char not in string.punctuation]
            s = ''.join(panc)
            words = nltk.word_tokenize(s)
            s = [w for w in words if w not in stopWords ]
            s = ' '.join(s)

            new_data.append([s,target[i]])
            #new_data.append([s,target[i].astype("float64")])


    new_data = np.array(new_data)

    #print(new_data[:,1])

    features = cv.fit_transform(new_data[:,0])
    target = np.array([tar for tar in new_data[:,1]])

    return new_data,features,target

def word_processer(text):
    
    data,_,_ = preprocesses()
    sent = nltk.sent_tokenize(text)

    cv = CountVectorizer()
    fit_text = cv.fit(data[:,0])
    
    data = cv.transform(sent)
    return data


def train2():
    _,x,y = preprocesses()
    x_train,x_test,y_train,y_test = train_test_split(x,y)

    network = MLPRegressor(verbose=True,max_iter=500,
            learning_rate='constant',learning_rate_init=0.01)

    network.fit(x_train,y_train)

    file_name = "neural_model_v2.pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(network, file)
    
    predict = network.predict(x_test)

    print(r2_score(y_test,predict,multioutput="raw_values"))



def train():

    _,x,y = preprocesses()

    x_train,x_test,y_train,y_test = train_test_split(x,y)

    svr = SVR(gamma="alpha")
    regr = MultiOutputRegressor(svr)
    regr.fit(x_train,y_train)

    """

    file_name = "model_v1.pkl"

    with open(file_name, 'wb') as file:
        pickle.dump(regr, file) """
    
    predict = regr.predict(x_test)

    print(r2_score(y_test,predict,multioutput="raw_values"))

    """
    print("##################################3")
    #acc = accuracy_score(y_train[0],predicted)
    #print(predicted)
    #print(y_train[0]) """

    

def main():
    #train2()
    preprocesses()
    #print("########################3")
    #print("word process",word_processer('This is rob is the main thing in the world'))

if __name__ == "__main__":
    main()
