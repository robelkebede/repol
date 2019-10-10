
# -*- coding:  utf-8 -*-
import time
from googletrans import Translator


def eng_to_amh(text):

    translator = Translator()

    init = time.time()
    data = translator.translate(text, dest='en')
    fin = time.time()

    delta = fin-init #time took

    return delta,data.text

def main():
    delta,data = eng_to_amh('0')

    print([data,delta])

if __name__ == '__main__':
    main()
