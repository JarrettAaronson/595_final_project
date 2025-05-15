import pandas as pd

def load_twitter_train(path='data/twitter_training.csv'):
    names = ['id','channel','sentiment','text']
    df = pd.read_csv(path, header=None, names=names, encoding='latin-1')
    return df[['text','sentiment']]

def load_twitter_val(path='data/twitter_validation.csv'):
    names = ['id','channel','sentiment','text']
    df = pd.read_csv(path, header=None, names=names, encoding='latin-1')
    return df[['text','sentiment']]
