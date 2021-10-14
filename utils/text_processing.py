import os, re, sys
from nltk.corpus import stopwords
import string
import spacy

def umlauts(text):
    """
    Replace umlauts for a given text
    
    :param word: text as string
    :return: manipulated text as str
    """
    
    tempVar = text # local variable
    
    # Using str.replace() 
    
    tempVar = tempVar.replace('ä', 'ae')
    tempVar = tempVar.replace('ö', 'oe')
    tempVar = tempVar.replace('ü', 'ue')
    tempVar = tempVar.replace('Ä', 'Ae')
    tempVar = tempVar.replace('Ö', 'Oe')
    tempVar = tempVar.replace('Ü', 'Ue')
    tempVar = tempVar.replace('ß', 'ss')
    
    return tempVar

def lemmatizer(text): 
    """
    Lemmetize words using spacy
    :param: text as string
    :return: lemmetized text as string
    """

    # global model_de
    doc = model_de(text)
    return [word.lemma_ for word in doc if len(word)>1]

def process(text, german_stop_words_to_use):
    """
    Pre-process texts with the following steps:
    1. Replace umlauts and special characters
    2. Replace punctuation with spaces
    3. Remove digits
    4. Remove german stopwords, according to the nltk dictionary
    5. Lemmatize words based on the German spacy model


    :param: text as string
    :german_stop_words_to_use: german stopwords as list
    :return: list of tokens
    """

    # text_lower = text.lower() German words don't have to be lowered, because uppercase is a syntactic feature
    remove_pun = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    remove_digits = str.maketrans('', '', string.digits)

    text_wo_umlauts = umlauts(text)
    text_wo_pun = text_wo_umlauts.translate(remove_pun)
    text_wo_num = text_wo_pun.translate(remove_digits)
    text_wo_stop_words = ' '.join([word for word in text_wo_num.split() if word.lower() not in german_stop_words_to_use])
    text_lemmatized = lemmatizer(text_wo_stop_words)
    

    return text_lemmatized

def clean_texts(texts):
    german_stop_words = stopwords.words('german')
    german_stop_words_to_use = []
    for word in german_stop_words:
        german_stop_words_to_use.append(umlauts(word))
    global model_de
    model_de = spacy.load('de_core_news_sm')
    return process(texts, german_stop_words_to_use)
