import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk

# ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
tokenizer  = RegexpTokenizer(r'[A-Za-z]+')

def clean_text(text):
    # if text is missing or not a str, return empty
    if not isinstance(text, str):
        return ""
    # lowercase and strip URLs/mentions
    text = re.sub(r"http\S+|@\S+", "", text.lower())
    # tokenize and remove stopwords
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)
