import pandas as pd
import streamlit as st
import numpy as np


def Get_random_games(data,
                tags_list) -> pd.DataFrame:
    ''' 
    每个tag回传两个
    '''
    game_data_df = data[data['median_playtime']!= 0 ]
    output_data = pd.DataFrame()
    # df = game_data_df.sort_values(by = 'steamspy_tags',ascending=True,inplace=False)
    for tag in tags_list: #tag is a string
        game_data_df['with_tag'] = game_data_df['steamspy_tags'].apply(lambda x: 1 if tag in str(x).lower() else 0)
        filtered_data = game_data_df[game_data_df['with_tag'] == 1]
        output_slice = filtered_data.sample(frac=1).reset_index(drop=False)#.head(2)
        output_data = pd.concat([output_data,output_slice])

    return output_data

def Get_data_by_tag(data,
                tag) -> pd.DataFrame:
    ''' 
    get data according to tag(string)
    '''
    # game_data_df = data[data['median_playtime']!= 0 ]
    game_data_df = data
    
    game_data_df['with_tag'] = game_data_df['tags'].apply(lambda x: 1 if str(tag).lower() in str(x).lower() else 0)
    filtered_data = game_data_df[game_data_df['with_tag'] == 1]

    return filtered_data

from collections import Counter
def find_most_common_subset(sets, threshold=None):
    if not sets:
        return set()
    
    all_elements = [elem for s in sets for elem in s]
    element_counts = Counter(all_elements)
    
    if threshold is None:
        threshold = len(sets) #// 2  # 可以调整这个阈值
    
    common_elements = {elem for elem, count in element_counts.items() if count >= threshold}
    result = list(common_elements)
    return result

def Get_tag_idxes(data,
                tags_list) -> dict:
    tag_idxes = {}
    tag_idx_list = []
    for tag in tags_list:
        a = Get_data_by_tag(data,tag)
        tag_idx = list(a['appid'].unique())
        tag_idxes[tag] = tag_idx
        tag_idx_list.append(tag_idx)
    return tag_idx_list

def Get_cooccur_games(data,tags_list):
    tag_idx_list = Get_tag_idxes(data,tags_list)
    common_idx = find_most_common_subset(tag_idx_list,None)
    return data[data['appid'].isin(list(common_idx))]

def Get_tag_slice(data,tags_list):
    tag_slice = Get_cooccur_games(data,tags_list)
    return tag_slice

def Table_tag_proportion(data,all_tags):
    ''' 
    return a dataframe
    three columns: tag, count, proportion
    '''
    #先算一个tag-比例-rank的表，再取出表中的rank
    count_tag = []
    for tag in all_tags:
        count_tag.append(Get_data_by_tag(data,tag).shape[0])
    rank_tag = pd.DataFrame({'tag':all_tags,
                             'count':count_tag})
    rank_tag['proportion'] = round(rank_tag['count']/data.shape[0],2)
    rank_tag = rank_tag.sort_values(by = 'proportion',ascending = False).reset_index()
    return rank_tag


def Calculate_tags_proportion(data,tags_list) -> float:
    len1 = Get_tag_slice(data,tags_list).shape[0]
    len2 = data.shape[0]
    return round(len1/len2,2)

def Calculate_tag_playtime(data,tags_list) -> float:
    slice = Get_tag_slice(data,tags_list)
    return round(np.mean(slice['median_playtime']),1)


import pandas as pd
from itertools import combinations

def count_tag_co_occurrences(data):
    co_occurrences = {}
    for tags in list(data['steamspy_tags']):
        tag_list = tags.split(';')
        for tag1, tag2 in combinations(set(tag_list), 2):
            pair = tuple(sorted([tag1, tag2]))
            if pair in co_occurrences:
                co_occurrences[pair] += 1
            else:
                co_occurrences[pair] = 1
    return co_occurrences

def get_top_co_occurring_tags(data, target_tag, top_n=30):
    co_occurrence_counts = count_tag_co_occurrences(data)
    tag_counts = {}
    for (tag1, tag2), count in co_occurrence_counts.items():
        if tag1 == target_tag:
            tag_counts[tag2] = tag_counts.get(tag2, 0) + count
        elif tag2 == target_tag:
            tag_counts[tag1] = tag_counts.get(tag1, 0) + count
    
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_tags[:top_n]