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
import train
import matplotlib.pyplot as plt




_,x,y = train.preprocesses()
x_train,x_test,y_train,y_test = train_test_split(x,y)




file_name = "./model/neural_model_v2.pkl"

with open(file_name, 'rb') as file:
    Pickled_LR_Model = pickle.load(file)



#print(r2_score(y_test,predict,multioutput="raw_values"))

print("training loss ",Pickled_LR_Model.score(x_train, y_train))
print("test loss ",Pickled_LR_Model.score(x_test, y_test))

objects = ('training loss', 'test loss')
y_pos = np.arange(len(objects))
x_pos = [Pickled_LR_Model.score(x_train, y_train), Pickled_LR_Model.score(x_test, y_test)]

bar =plt.bar(y_pos, x_pos, align='center', alpha=0.4)
bar[0].set_color('b')
bar[1].set_color('r')

plt.xticks(y_pos, objects)
plt.yticks(np.arange(1, step=0.02))
plt.ylabel('num_reaction')
#plt.title('Social Media Reactions. accuracy is {}'.format(round(pre*100),4))

plt.show() 

