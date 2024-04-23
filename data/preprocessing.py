import pandas as pd 
import string 
from nltk.corpus import stopwords
import re 


def preprocess(text):

    punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation

    stop_words = stopwords.words('arabic')

    arabic_diacritics = re.compile("""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

    #remove punctuations
    translator = str.maketrans('', '', punctuations)
    text = text.translate(translator)

    # remove Tashkeel
    text = re.sub(arabic_diacritics, '', text)

#     #stemming
#     st = ISRIStemmer()
#     stemmed_words = []
#     words = nltk.word_tokenize(text)
#     for w in words:
#         stemmed_words.append(st.stem(w))
#     text = " ".join(stemmed_words)

    #remove longation
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)

    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text


def clean_df(df):
    
    df.drop_duplicates(subset=['request'], keep='first', inplace=True)

    df['request'] = df['request'].apply(preprocess)
    
    df['encoded_intent'] = df['intent'].astype('category').cat.codes
    
    return df