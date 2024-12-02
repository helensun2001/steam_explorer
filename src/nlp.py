import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pylab import xticks, np
import re  

import spacy
from spacy.matcher import PhraseMatcher
from spacy.language import Language
from gensim.models import KeyedVectors
import requests

stopp = 'data/stop_words_steam.txt'

def get_game_details(app_id):
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data[str(app_id)]["success"]:
            game_data = data[str(app_id)]["data"]
            return  {
                "name": game_data.get("name"),
                # "detailed_description": re.sub(r'<[^>]+>', ' ', game_data.get("detailed_description")).replace('  ',' '),
                "detailed_description":game_data.get("detailed_description"),
                # "original":game_data.get("detailed_description"),
                "header_image": game_data.get("header_image"),
                "screenshots": game_data.get("screenshots"),
                "short_description": game_data.get("short_description"),
                # "positive_reviews": game_data.get("recommendations", {}).get("total", 0),
                "tags": [tag['description'] for tag in game_data.get("genres", [])],
                "developer":game_data.get("developers")[0],
                "release_date":game_data.get("release_date")['date']
            }

        else:
            return None
    else:
        return None
def compare_input_genres(games,input_wordbags,model):
    dic = {}
    for i in range(0,games.shape[0]):
        compare_game_des = games.iloc[i]['steamspy_tags']
        # print(games.iloc[i]['name'])
        compare_wordbags = compare_game_des.lower().split(';')
        # print(compare_wordbags)
        similarity = calculate_similarity(input_wordbags,compare_wordbags,model)
        # print(similarity)

        dic[games.iloc[i]['name']] = similarity
    return dic

def compare_input_games(games,input_wordbags,model):
    dic = {}
    for i in range(0,games.shape[0]):
        compare_game_des = games.iloc[i]['wordbag']
        compare_wordbags = str(compare_game_des).split(',')
        
        # compare_wordbags = Clean_texts([compare_game_des],stop_path=stopp,phrase_path=False)[0]
        # print(compare_wordbags)
        similarity = calculate_similarity(input_wordbags,compare_wordbags,model)
        # print(similarity)

        dic[games.iloc[i]['name']] = similarity
    return dic

def rank_dict(data, n=10):

    if not isinstance(data, dict):
        raise ValueError("输入数据必须是字典类型")

    # 按值从大到小排序，并取前N名
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:n]

    # 转换为数据框
    df = pd.DataFrame(sorted_items, columns=['name', 'Value'])
    
    return df

def load_glove_embeddings(filepath):
    """加载 GloVe 嵌入"""
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings



def get_vectors(keywords, model):
    """获取关键词的词向量列表，跳过不存在的单词"""
    vectors = []
    for word in keywords:
        if word in model:  # 只添加存在于 GloVe 词典的单词
            vectors.append(model[word])
    return vectors

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(keywords_set1, keywords_set2, model):
    """计算两个关键词集合的余弦相似度"""
    def average_vector(vectors):
        if len(vectors) == 0:
            return np.zeros(len(next(iter(model.values()))))  # 取模型中一个向量的维度
        return np.mean(vectors, axis=0)
    
    vectors1 = get_vectors(keywords_set1, model)
    vectors2 = get_vectors(keywords_set2, model)
    
    avg_vector1 = average_vector(vectors1)
    avg_vector2 = average_vector(vectors2)
    
    # 计算余弦相似度
    similarity = cosine_similarity([avg_vector1], [avg_vector2])[0][0]
    return similarity


def Get_data_by_tag(data,
                tag) -> pd.DataFrame:
    ''' 
    get data according to tag(string)
    '''
    # game_data_df = data[data['median_playtime']!= 0 ]
    game_data_df = data
    
    game_data_df['with_tag'] = game_data_df['steamspy_tags'].apply(lambda x: 1 if str(tag).lower() in str(x).lower() else 0)
    filtered_data = game_data_df[game_data_df['with_tag'] == 1]

    return filtered_data

def load_wordvec(path):
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    return model

# 读取停用词
def load_stopwords(file_path):
    with open(file_path, "r") as file:
        stopwords = [line.strip() for line in file]
    return stopwords


def Lemmatize_text(text,nlp):
    """ 
    Args:
        text(string): a text string, the doc unit
        nlp(object): nlp model, with predefined model from spacy, matcher added

    Returns: 
        list: lower-cased keywords of each text wrapped in a list

    """
    
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]

def load_spacy_phrasematcher(phrase_path):
    with open(phrase_path, "r",encoding = "utf-8") as f:
        phrases = f.readlines()
        phrase_list = [phrase.strip() for phrase in phrases]

        
    nlp = spacy.load("en_core_web_sm")

    matcher = PhraseMatcher(nlp.vocab)

    # 添加自定义词组
    # phrases = ["President Biden", "new york", "artificial intelligence"]
    patterns = [nlp(text) for text in phrase_list]
    matcher.add("SpecialPhrases", patterns)

    @Language.component("merge_phrases")
    def merge_phrases(doc):
        matches = matcher(doc)
        with doc.retokenize() as retokenizer:
            for _, start, end in matches:
                retokenizer.merge(doc[start:end])
        return doc

    # 将自定义组件添加到pipeline
    nlp.add_pipe("merge_phrases")#, after="ner")
    return nlp

def text_to_tokens(text,nlp):
    tokens = []
    doc = nlp(text)
    for token in doc:
        x = re.sub(r'[^a-zA-Z\s]', '',token.lemma_.lower())
        if x != "" and len(x)>2:  # remove word with 2 or less letters
            tokens.append(re.sub(r'br$', '', x))#re.sub(r'[^a-zA-Z\s]', '', token.lemma_.lower())
    return tokens


def Clean_texts(input_texts:list,stop_path,phrase_path):
    """ 
    Args:
        input_texts(list): all texts wrapped in a list
        stop_path(string): path of stop words
        phrase_path(bool | string): set to False if no pre-defined phrase.txt; set to doc path if some phrases should not be splitted

    Returns: 
        list: keywords of each text wrapped in a list

    """
    tokens_list = []
    stopwords = load_stopwords(stop_path)
        
    nlp = spacy.load("en_core_web_sm")
    
    for sentence in input_texts:
        sentence = re.sub(r'[0-9]', '', str(sentence))
        lemmatized = Lemmatize_text(sentence,nlp)

        cleaned_sentence = []
        for word in lemmatized:  
            if (word in stopwords) is False:   # remove stop words
                cleaned_sentence.append(word)
        new_sentence = ' '.join(cleaned_sentence)

        if phrase_path != False:
            nlp = load_spacy_phrasematcher(phrase_path)
        nlp.max_length = 10000
        tokens = text_to_tokens(new_sentence,nlp)
        
        tokens_list.append(tokens)

    return tokens_list

def get_vectors(keywords, model):
    vectors = []
    for word in keywords:
        if word in model:
            vectors.append(model[word])
    return vectors
