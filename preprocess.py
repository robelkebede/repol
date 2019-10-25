
import pandas as pd
from translate import eng_to_amh
import numpy as np
from tqdm import tqdm

fana_data = pd.read_csv("fana_data.csv")
the_data = fana_data.filter(["text","reactions","likes","ahah","love","wow","sigh","grrr","comments"]).fillna(0)


#k to thousends
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
    for index,row in tqdm(d.iterrows()):
        if index>987:
            if row["text"] != 0:
                _,translated_text = eng_to_amh(row["text"][0:2000])
                
                final_data.append([translated_text,k2t(row["reactions"]),k2t(row["likes"]),
                    k2t(row["ahah"]),k2t(row["love"]),k2t(row["wow"]),
                    k2t(row["sigh"]),k2t(row["grrr"]),k2t(row["comments"])])
                print("INDEX ",index)

                if index == 1100:
                    np.save("fana_news_data_{}".format(index),final_data)
                    print("DATA_SAVED")

            
            else:
                print("PASS")


    return final_data



def final_data():
    data = np.array(processes_data(the_data))
    
def main():

    #print(the_data)
    final_data()

    #print(the_data["text"][2][0:2001])

    #data = np.load("fana_news_data.npy",allow_pickle=True)
    #print(data)
    


if __name__ =="__main__":
    main()

