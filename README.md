
# Repol

I trained a multi-output linear regression model that predicts Ethiopian Facebook users reaction to a political news post 

writing a web crawler for facebook is hard so i used https://github.com/rugantio/fbcrawl

I scraped data from a Facebook Ethiopian news pages

I used googletrans a free unofficial google trainslate library to convert from Amahric to English

## Warning 
 <p style="color:red;">the data that is use to train this model might be biased you can see the data i used to train this model in /data  and the unprocessed data in /dataset </p>

## Requirements

python 3.x

## Installation

```bash 
git clone https://github.com/robelkebede44/repol.git

pip install -r requirements.txt
```

## Usage

``` bash
cd ./repol
python repol.py --text="Prime Minster Dr. Abiy Ahmed, launched the ambitious Green Legacy campaign that set a milestone to plant 200 million tree seedlings"
```

![Alt text](Figure_1.png?raw=true "Reactions")


## Contributing

Pull request are welcome. specially new data crawlerd from diffrient Facebook news pages and please open an issue first 


## License

[MIT] (https://choosealicense.com/licenses/mit/)



