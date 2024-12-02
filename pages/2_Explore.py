import streamlit as st
import pandas as pd
import csv
import streamlit.components.v1 as components
from src.data_processing import *
from src.visualization import *
from src.visualization import plot_publisher_barchart

def Read_csv_to_list(path):
    game_tags = []
    with open(path, 'r') as file:
        for row in file:
            game_tags.append(str(row).strip('\n'))
    return game_tags

st.set_page_config(page_title="Game Explore", page_icon="📊")

st.markdown("# Game Explore")
st.sidebar.header("Game Explore")

# Load your data
data = pd.read_csv("data/steam.csv")
data['release_date'] = pd.to_datetime(data['release_date'])
game_tags = Read_csv_to_list("data/tags.csv")
extended_tags = game_tags + ['All']


# 初始化session state
if 'selected_tags' not in st.session_state:
    st.session_state.selected_tags = []

# 多选框
selected_games = st.multiselect(
    "Select game tags:",
    options=extended_tags,
    default=st.session_state.selected_tags,
    key="game_multiselect"
)

# 显示选中的游戏标签
# st.write("已选择的游戏标签:", ", ".join(st.session_state.selected_tags))

# 获取所有发行商
publishers = ['All'] + data['publisher'].unique().tolist()
# 发行商选择框
selected_publisher = st.selectbox("Select a publisher:", publishers)

# 同步多选框和session state
if selected_games != st.session_state.selected_tags:
    st.session_state.selected_tags = selected_games

# 添加"探索"按钮
if st.button("EXLPORE"):

    # st.write(f"You've selected {' + '.join(st.session_state.selected_tags)} games")
    if selected_publisher == 'All':
        filtered_data = data.copy()
        if 'All' in selected_games:
            tag_slice = data.copy()
        else:
            st.session_state.selected_tags = selected_games
            tag_slice = Get_tag_slice(data,st.session_state.selected_tags)
    else:
        filtered_data = data[data['publisher'] == selected_publisher]
        if 'All' in selected_games:
            tag_slice = filtered_data.copy()
        else:
            st.session_state.selected_tags = selected_games
            tag_slice = Get_tag_slice(filtered_data,st.session_state.selected_tags)
    

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_metric('Number of games found',f'{tag_slice.shape[0]}')
    with col2:
        median_playtime_selected = round(np.mean(tag_slice[tag_slice['median_playtime']>0]['median_playtime']),1)
        display_metric('Average playtime (hours)',f'{median_playtime_selected} H')
    with col3:
        tag_slice['rating_ratio'] = tag_slice['positive_ratings']*100/(tag_slice['positive_ratings']+tag_slice['negative_ratings'])
        average_positive_ratings = round(np.mean(tag_slice['rating_ratio']),2)
        display_metric('Average positive ratings',f'{int(average_positive_ratings)} %')
    with col4:
        average_price_selected = round(np.mean(tag_slice['price']),1)
        display_metric('Average price',f'{average_price_selected} $')


    if tag_slice.shape[0] == 0:
        st.write('Wow! you have discovered a rare combo!')
    
    elif 'All' in selected_games: #没有选标签
        
        plot_count_trend(tag_slice)
        plot_time_trend(tag_slice,'price','average_price')

        # plot distribution of developers
    
    else: #选了1-2个标签
        # Display_tag_analysis(filtered_data,st.session_state.selected_tags,game_tags)
        # color = '#87CEEB'
        # highlight_value = Calculate_tag_playtime(tag_slice,st.session_state.selected_tags)
        # highlight_color =  '#FF0000'
        # plot_distribution(data[data['median_playtime']!=0],'median_playtime',  color, highlight_value, highlight_color)

        # st.dataframe(tag_slice)
        st.write('\n\n\n')
        plot_mean_playtime_by_owners(tag_slice)
        plot_publisher_barchart(tag_slice)

        # plot_tag_network(tag_slice)

        

        plot_count_trend(tag_slice)

        plot_time_trend(tag_slice,'price','average_price')

        plot_time_trend(tag_slice,'rating_ratio','positive ratings')

        
        
        
        
        

    
    
    

    # create graph based on the first input tag
    # node_trace, edge_trace = Create_network_graph(top_tags, st.session_state.selected_tags[0])
    # Plot_cooccur_graph(node_trace,edge_trace)
    # 创建网络图
    
    
    

        



# # Create visualizations
# genre_chart = create_genre_chart(data)  # Implement this function in visualization.py
# rating_chart = create_rating_chart(data)  # Implement this function in visualization.py

# st.altair_chart(genre_chart, use_container_width=True)
# st.altair_chart(rating_chart, use_container_width=True)