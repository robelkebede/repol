
import pandas as pd
from translate import eng_to_amh
import numpy as np
from tqdm import tqdm

fana_data = pd.read_csv("./dataset/fana_data.csv")
the_data = fana_data.filter(["text","reactions","likes","ahah","love","wow","sigh","grrr","comments"]).fillna(0)


#k to thousends eg (2.5k to 2500)
def k2t(d):
    d =str(d)
    if d[-1] == "K":
        digit = d[0:-1]
        extracted = float(digit)*1000
        return  extracted
    else:
        return d

def processes_data(d):
    final_data = []
    start = 987
    end = 1100
    for index,row in tqdm(d.iterrows()):
        #processes data in segment because googletrans crash after processing 50-40 text
        if index>start:
            if row["text"] != 0:
                _,translated_text = eng_to_amh(row["text"][0:2000])
                
                final_data.append([translated_text,k2t(row["reactions"]),k2t(row["likes"]),
                    k2t(row["ahah"]),k2t(row["love"]),k2t(row["wow"]),
                    k2t(row["sigh"]),k2t(row["grrr"]),k2t(row["comments"])])
                print("INDEX ",index)

                if index == end:
                    np.save("fana_news_data_{}".format(index),final_data)
                    print("DATA_SAVED")

            else:
                print("PASS")

    return final_data



def final_data():
    data = np.array(processes_data(the_data))
    #save data.npy
    
def main():
    final_data()

if __name__ =="__main__":
    main()

