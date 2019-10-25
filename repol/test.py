
import numpy as np
import matplotlib.pyplot as plt
from train import preprocesses
"""
data0 = np.load("fana_news_data_50.npy")
data1 = np.load("fana_news_data_100.npy")
data2 = np.load("fana_news_data_150.npy")
data3 = np.load("fana_news_data_200.npy")
data4 = np.load("fana_news_data_250.npy")
data5 = np.load("fana_news_data_300.npy")
data6 = np.load("fana_news_data_350.npy")
data7 = np.load("fana_news_data_400.npy")
data8 = np.load("fana_news_data_450.npy")
data9 = np.load("fana_news_data_500.npy")
data10 = np.load("fana_news_data_550.npy")
data11 = np.load("fana_news_data_600.npy")
data12 = np.load("fana_news_data_700.npy")
data = np.concatenate([data0,data1,data2,data3,data4,data5,data6,data7,data8,data9,
        data10,data11,data12]) 

"""
def data_preprocess_float():

    data = np.load("fana_data_700_new.npy")

    feature = data[:,0]
    target = data[:,1:9]

    final_target = []

    for t in target:
        me = [float(x) for x in t]
        final_target.append(me)

    #data = np.array([feature,target]).reshape(feature.shape[0],8)
    data = np.array([feature,final_target])

    np.save("700.npy",data)
    print("SAVED")


data = np.load("700.npy",allow_pickle=True)

print(data[0][500])
print(data[1][500])
