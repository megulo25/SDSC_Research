# Explainable AI Research:

## How to train the keras model:
Make sure there's you extract the nabirds dataset into a folder in ../SDSC_Research/TRAIN/data

Go into ../SDSC_Research/TRAIN and run train_keras.py
    
    `python train_keras.py 0`

    The '0' represents the gpu id that you want to train your model on.

## How to run scraper
In mongodb create a database called 'Birds'. Then create a collection within 'Birds' called 'bird_info'. Each document collected will be stored in 'bird_info'.

Go in ../SDSC_Research/bird_scraper and run bird_scraper.py

    `python bird_scraper.py`

    It should automatically connect with your database and store the documents.