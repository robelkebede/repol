
import pandas as pd
from translate import eng_to_amh
import numpy as np

fana_data = pd.read_csv("fana_data.csv")
the_data = fana_data.filter(["text","reactions","likes","ahah","love","wow","sigh","grrr","comments"]).fillna(0)

#TODO add one additional feature (num of COMMENTS)

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
    for index,row in d.iterrows():
        
        if row["text"] != 0:
            _,translated_text = eng_to_amh(row["text"])
            
            final_data.append([translated_text,k2t(row["reactions"]),k2t(row["likes"]),
                k2t(row["ahah"]),k2t(row["love"]),k2t(row["wow"]),
                k2t(row["sigh"]),k2t(row["grrr"]),k2t(row["comments"])])
            print(translated_text)
        else:
            print("PASS")


    return final_data



def final_data():
    data = np.array(processes_data(the_data))
    np.save("fana_data_test222.npy",data)
    print("DATA SAVED")


def main():
    final_data()
    #processes_data(the_data)
    #print(k2t(234))
    #data = np.load("fana_data_test.npy",allow_pickle=True)
    #d = k_to_thousend(data)
    


if __name__ =="__main__":
    main()

#delta,data = eng_to_amh(d)
