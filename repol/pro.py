
import pandas as pd
from translate import eng_to_amh
import numpy as np

fana_data = pd.read_csv("fana_data.csv")
the_data = fana_data.filter(["text","reactions","likes","ahah","love","wow","sigh","grrr"]).fillna(0)

#TODO add one additional feature (num of COMMENTS)


def processes_data(d):
    final_data = []
    for index,row in d.iterrows():
        
        if row["text"] != 0:
            _,translated_text = eng_to_amh(row["text"])
            
            final_data.append([translated_text,row["reactions"],row["likes"],
                row["ahah"],row["love"],row["wow"],
                row["sigh"],row["grrr"]])
        else:
            print("PASS")


    return final_data


def k_to_thousend(data):

    data = data[:,1:7]

    for d in data:
        
        for ed in d:

            if ed[-1] == "K":
                digit = ed[0:-1]
                extracted = float(digit)*1000
                print(["spcial",ed,extracted])
                #ed[-1] = str(extracted)
    #must b float(data_2)
    #change all data in the array to float
    #mprint(data_2)



def final_data():
    data = np.array(processes_data(the_data))
    #k_to_thousend(data)
    np.save("fana_data_test.npy",data)
    print("DATA SAVED")


def main():
    #final_data()
    data = np.load("fana_data_test.npy",allow_pickle=True)
    d = k_to_thousend(data)

    print(d)

    #print(data[:,1:7])
    


if __name__ =="__main__":
    main()

#delta,data = eng_to_amh(d)
