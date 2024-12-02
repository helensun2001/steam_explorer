import pandas as pd
import streamlit as st
from src.nlp import *
import time
import functools
from src.visualization import *

def compute_with_loading(func):
    """
    A decorator to measure the execution time of a function and display a loading GIF while it runs.
    
    Args:
        func (function): The function to measure and wrap.
    
    Returns:
        function: A wrapped function with loading indicator and time measurement.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Display a loading GIF
        with st.spinner("Processing..."):
            start_time = time.time()  # Start timer
            result = func(*args, **kwargs)  # Execute the function
            end_time = time.time()  # End timer
            
        # Display the execution time
        st.success(f"Execution completed in {end_time - start_time:.2f} seconds.")
        return result
    
    return wrapper

# 加载 FastText 模型
# model = fasttext.load_model('cc.en.50.bin')

# api
def Read_csv_to_list(path):
    game_tags = []
    with open(path, 'r') as file:
        for row in file:
            game_tags.append(str(row).strip('\n'))
    return game_tags


game_tags = Read_csv_to_list("data/tags_for_creation.csv")
games = pd.read_csv("data/games_to_recommend.csv")
# model = load_wordvec(path = 'data/GoogleNews-vectors-negative300.bin.gz')
# Load the word embedding model 50d - GloVe as an example
glove_model = load_glove_embeddings("data/glove.6B.50d.txt")
# 定义页面导航
if "page" not in st.session_state:
    st.session_state.page = "create"  # 默认页面为 "create"

def go_to_page(page_name):
    st.session_state.page = page_name


# @compute_with_loading
@st.cache_data
def get_top10_similar_games(games, input_genre,input_description,rank_method,model):
    input_wordbags = Clean_texts([input_description],stop_path=stopp,phrase_path=False)[0]
    
    if input_genre == 'Not Sure':
        st.caption('This search may take longer time😯 Tons of games being screened...')
        game_similarity_top10 = rank_dict(compare_input_games(games,input_wordbags,model),n = 15)
        result = pd.merge(game_similarity_top10,games[['appid','name']],how = 'inner',on = 'name')
        with st.expander("See recommendation in tables"):
            st.write(game_similarity_top10.reset_index(drop=True))
        return result
    
    else:
        games['input_check'] = games['tags'].apply(lambda x: 1 if str(input_genre).lower() in str(x).lower() else 0 )
        games_with_same_tag = games[games['input_check']==1]
    # st.write(games_with_same_tag)
        first_round_recall_cnt = games['input_check'].sum()
        print(first_round_recall_cnt) 
        if first_round_recall_cnt == 0:
            st.write('The genre does not has any game within your selected price range.')
            return None
        else:
            if first_round_recall_cnt <= 10:
                # top_des = pd.merge(games_with_same_tag,games[['name','wordbag']],how = 'inner',on = 'name')
                game_similarity_top10 = rank_dict(compare_input_games(games_with_same_tag,input_wordbags,model))
            
            else:
                if rank_method == "Random": #随机抽样1/4的游戏
                    data_slice = games_with_same_tag.sample(first_round_recall_cnt//4) 

                if rank_method == "Most Popular":
                    data_slice = games_with_same_tag.sort_values(by = ['owner_level','positive_ratio'],ascending = [False,False]).head(first_round_recall_cnt//4)

                if rank_method == "Less Discovered":
                    data_slice = games_with_same_tag.sort_values(by = ['owner_level','positive_ratio'],ascending = [True,False]).head(first_round_recall_cnt//4)
                
                compare_tag_and_des = compare_input_games(data_slice,input_wordbags,model)
                tag_similarity_topn = rank_dict(compare_tag_and_des,n = first_round_recall_cnt//4)

                top10_des = pd.merge(tag_similarity_topn,games[['name','wordbag']],how = 'inner',on = 'name')
                game_similarity_top10 = rank_dict(compare_input_games(top10_des,input_wordbags,model),n = min(15,first_round_recall_cnt//4))
            
            result = pd.merge(game_similarity_top10,games[['appid','name']],how = 'inner',on = 'name')

            with st.expander("See recommendation in tables"):
                st.write(game_similarity_top10.reset_index(drop=True))
            st.caption("Currently the recommendation result displays no more than 15 games. Fewer games will be presented if you select a rare tag with less than 60 games among Steam's history.")
            # print(game_similarity_top10)
            return result

@compute_with_loading
def display_recommended_games2(recommended_game_ids):
    for app_id in recommended_game_ids:
        i = 0
        # 假设 get_game_details 是一个函数，返回游戏详细信息
        game_details = get_game_details(app_id)
        if i<= 10:
            if game_details is not None:
                # 使用 expander 包含图片、基本信息和详细信息
                
                # 撑满网页的图片
                st.image(game_details["header_image"], use_container_width=True)

                # 分为三栏显示基本信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    link = f"https://store.steampowered.com/app/{app_id}"
                    gamename = f"{game_details['name']}"
                    st.markdown(f"**Name:** [{gamename}]({link})")
                    # st.markdown(f"[{gamename.split(':')[1]}]({link})")
                    # st.markdown([点击这里访问 Streamlit 官网](https://streamlit.io)")
                with col2:
                    st.markdown(f"**Developer:** {game_details['developer']}")
                with col3:
                    st.markdown(f"**Release Date:** {game_details['release_date']}")

                # 显示 Tags 和短简介
                st.markdown(f"**Tags:** {', '.join(game_details['tags'])}")
                st.markdown(f"**Short Description:** {game_details['short_description']}")

                # 展开显示长简介
                with st.expander("Show Full Description"):
                    st.markdown(game_details["detailed_description"], unsafe_allow_html=True)

                # 添加分隔线
                st.divider()
                i+=1





def filter_games_by_price(price_range,games):
    min_price, max_price = price_range[0],price_range[1]
    filtered_games = games[games['price'] <= max_price]
    if filtered_games.shape[0] == 0:
        st.write('max price is too low')
    filtered_games = filtered_games[filtered_games['price'] >= min_price]
    if filtered_games.shape[0] == 0:
        st.write('min price is too high')
    return filtered_games


# 主页面 "create"
if st.session_state.page == "create":
    st.title("🎮 Game Hunt")
    game_tags.append('Not Sure')

    input_genre = st.selectbox("Select a basic game genre to start:", game_tags)

    # 文本输入框
    input_description = st.text_area(
        "Describe your ideation for your game! \n (artistic style, gameplay, storyline, creative elements etc... ",
        max_chars=1500,  # 限制字符数（约等于 200 单词）
        height=150
    )
    rank_method = st.radio(
    "Choose a recommendation mode:",
    options=[
             "Most Popular", # (Prioritize the popularity and amount of owners of similar games)
             "Less Discovered", # (Prioritize the games owned by fewer people and are less discovered.)
             "Random"] # (Look for similar games randomly from the selected genre )
)
    # # 显示当前单词计数
    input_price_range = st.slider(
    "Select price range:",
    min_value=0,  # 最小值
    max_value=1000,  # 最大值
    value=(0, 150),  # 默认选定的区间
    step=1  # 调节的步长
)
    
    word_count = len(input_description.split())
    # st.write(f"Word Count: {word_count}/200")
    
    # 按钮跳转到新页面
    if st.button("Find me some good games!"):

        
        if word_count > 200:
            st.warning("The input exceeds 200 words. Please reduce the word count.")
        elif word_count == 0:
            st.warning("Please enter at least one word.")
        else:
            games_price_range = filter_games_by_price(input_price_range,games)
            game_similarity_top10 = get_top10_similar_games(games_price_range, input_genre,input_description,rank_method,glove_model)
           
            if game_similarity_top10 is None:
                st.write('Try another description!')
            else:
                recommended_games_for_chart = pd.merge(game_similarity_top10,games,how = 'inner',on = ['appid','name'])
                tab1, tab2 = st.tabs(["Game Details", "Game Charts"])
                with tab1:
                    st.title("The following similar games are found.")
            
                    display_recommended_games2(list(game_similarity_top10['appid']))

                    if st.button("Back"):
                        go_to_page("create")

                with tab2:
                    col1, col2, col3, col4 = st.columns(4)
                    tag_slice = recommended_games_for_chart.copy()
                    with col1:
                        display_metric('Number of games found',f'{tag_slice.shape[0]}')
                    with col2:
                        non_zero_playtime = [r for r in list(tag_slice['median_playtime']) if r > 0 ]
                        median_playtime_selected = round(np.mean(non_zero_playtime),1)
                        display_metric('Average playtime (hours)',f'{median_playtime_selected} H')
                    with col3:
                        tag_slice['rating_ratio'] = tag_slice['positive_ratings']*100/(tag_slice['positive_ratings']+tag_slice['negative_ratings'])
                        non_zero_ratings = [r for r in list(tag_slice['rating_ratio']) if r not in [-1,0]]
                        average_positive_ratings = round(np.mean(non_zero_ratings),2)
                        display_metric('Average positive ratings',f'{int(average_positive_ratings)} %')
                    with col4:
                        non_zero_prices = [r for r in list(tag_slice['price']) if r != -1]
                        average_price_selected = round(np.mean(non_zero_prices),1)
                        display_metric('Average price',f'{average_price_selected} $')
                    plot_mean_playtime_by_owners(recommended_games_for_chart)
                    plot_count_trend(recommended_games_for_chart)

                    plot_time_trend(recommended_games_for_chart,'price','average_price')

                    plot_time_trend(recommended_games_for_chart,'positive_ratio','positive ratings')
                    st.caption('Games with missing data in these variables are not displayed on the graphs.')
                    