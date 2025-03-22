#%% LIB
import pandas as pd

import nltk
from nltk import ngrams
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
nltk.download('stopwords')

from pyvi import ViTokenizer

from collections import Counter

from gensim import corpora
from gensim.models import LdaModel

import re
import string

#%% CLEAN DATA
# NGAYMOIONLINE DATA
df_ngaymoi = pd.read_json('database/ngaymoionline_data.json')
df_ngaymoi = df_ngaymoi.drop_duplicates(keep=False)

# DANTRI DATA

df_dantri = pd.read_json('database/dantri_data.json')
df_dantri = df_dantri.drop_duplicates(keep=False)
df_dantri = df_dantri[~df_dantri['text'].str.contains('--')]

# MANUAL DATA
df_manual = pd.read_json('database/ngaymoionline_data.json')
df_manual = df_manual.drop_duplicates(keep=False)

df = pd.concat([df_ngaymoi, df_dantri, df_manual])

text = " ".join(row for row in df['text'])

# Remove extra spacd
for _ in range(10):
    text = text.replace("  ", " ")

#%% UNSUPERVISED TOPIC DETECT W/T LATENT DIRECHLET ALLOCATION
documents = sent_tokenize(text)

# Stop word
with open("vietnamese-stopwords.txt", "r") as file:
    vn_stopwords = file.read().splitlines()

vn_stopwords = [word.replace(' ', '_') for word in vn_stopwords]
    
tokens = [  
    [word.lower() for word in ViTokenizer.tokenize(doc).split()
     if word not in vn_stopwords
     ] for doc in documents
    ]

def keep_meaningful_word(word):
    if word.isdigit():
        return False
    
    if re.fullmatch(r'\W+', word):
        return False
    
    if all(char in string.punctuation for char in word):
        return False
    
    return True

tokens_updated = [
    [w for w in token if keep_meaningful_word(w)]
    for token in tokens
    ]

all_words = []
for token in tokens_updated:
    all_words.extend(token)

counter_word = Counter(all_words)
top_common_word = [
    word[0] for word in counter_word.most_common(int(0.05*len(counter_word)))
    ]

(pd.Series(top_common_word[:100])
 .rename('top_100_most_common')
 .to_excel('top_common_words.xlsx')
 )

tokens_updated = [
    [word for word in token if word not in top_common_word] 
    for token in tokens_updated
    ]

tokens_updated = [token for token in tokens_updated if len(token)>=2]


dictionary = corpora.Dictionary(tokens_updated)

corpus = [dictionary.doc2bow(text) for text in tokens_updated]

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,
    passes=10,
    random_state=42
    )

topics = lda_model.print_topics(num_words=20)

topics_dict = {"top_weighted_words": []}

for topic in topics:
    topics_dict["top_weighted_words"].append(
        ", ".join(topic[1].split('"')[1::2])
        )
    
df_topics = pd.DataFrame(topics_dict)

df_topics.to_excel('topic_allocation.xlsx')
    
#%% N-GRAM ANALYSIS
def export_ngram_list(text="a sentence", num_gram=2, num_top=50):
    result = {"word": [],
              "frequency": []}
    
    tokens = text.lower().split()
    n_grams = list(ngrams(tokens, num_gram))
    counter_ngrams = Counter(n_grams)
    counter_ngrams_top = counter_ngrams.most_common(num_top)
    
    for i in range(num_top):
        word = " ".join(doc for doc in counter_ngrams_top[i][0])
        count = counter_ngrams_top[i][1]
        
        # Save in result
        result["word"].append(word)
        result["frequency"].append(count)
    
    df = pd.DataFrame(result)
    df.to_excel(f'{num_top} most common {num_gram}-gram words.xlsx')
    return df

export_ngram_list(text=text, num_gram=4, num_top=100)
    
