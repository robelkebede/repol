
# Repol

* I trained a multi-output linear regression model that predicts Ethiopian Facebook users reaction to a political news

* I scraped facebook using fbcrawl  (https://github.com/rugantio/fbcrawl)

* I scraped data FBC(Fana broadcasting conrporate) news Page

* I used googletrans a free unofficial google trainslate library to convert from Amahric to English

## Limitations 

* Very small Dataset
* Future Me: <font color="red"> THIS REPO HAS A LOT OF PROBLEMS</font>
* not normalized data 
* bad word representation 
* poor translation 

## Requirements

python 3.x

## Installation

```bash 
git clone https://github.com/robelkebede/repol.git

pip install -r requirements.txt
```

## Usage

``` bash
cd ./repol
python repol.py --text="Green Legacy campaign"
```

![Alt text](Figure_1.png?raw=true "Reactions")


## Training Loss


![Alt text](loss.png?raw=true "Loss")


## Contributing

Pull request are welcome. specially new data crawlerd from diffrient Facebook news pages and please open an issue first 


## License

[MIT] (https://choosealicense.com/licenses/mit/)



