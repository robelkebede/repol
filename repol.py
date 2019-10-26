import nltk
import joblib
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_extraction.text import CountVectorizer
import argparse,sys
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from train import preprocesses

   
def word_processer(text):
    
    data,_,_ = preprocesses()
    sent = nltk.sent_tokenize(text)

    cv = CountVectorizer()
    fit_text = cv.fit(data[:,0])                                                       #why
              
    data = cv.transform(sent)
    return data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--text',type=str
            ,help='input a sentence'
            ,default="This is a test")

    args = parser.parse_args()

    word = word_processer(args.text)

    file_name = "./model/neural_model_v2.pkl"

    with open(file_name, 'rb') as file:
      Pickled_LR_Model = pickle.load(file)
     
    #score = Pickled_LR_Model.score(x_test, y_test)
    pre = Pickled_LR_Model.predict(word) 

    objects = ('reaction', 'likes', 'ahah', 'Love', 'Wow', 'Sigh','Grrr','Comments')
    y_pos = np.arange(len(objects))
    x_pos = pre.reshape(-1)
    
    print(y_pos)
    print(x_pos)
    print("score ",pre)

    
    bar =plt.bar(y_pos, x_pos, align='center', alpha=0.5)
    bar[0].set_color('b')
    bar[1].set_color('b')
    bar[2].set_color('y')
    bar[3].set_color('r')
    bar[4].set_color('y')
    bar[5].set_color('y')
    bar[6].set_color('r')
    bar[7].set_color('b')
    plt.xticks(y_pos, objects)
    plt.ylabel('num_reaction')
    #plt.title('Social Media Reactions. accuracy is {}'.format(round(pre*100),4))

    plt.show() 

    
if __name__ == "__main__":
    main()
