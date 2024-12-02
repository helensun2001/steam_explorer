
import streamlit as st
import pandas as pd
from src.data_processing import Calculate_tags_proportion, Table_tag_proportion
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

from datetime import datetime
import plotly.graph_objects as go
import datetime as dt

def Display_tag_analysis(data,tags_list,all_tags):
    proportion = Calculate_tags_proportion(data,tags_list)
    st.write(f'Games with this tag account for {proportion*100}% of all historical steam games.')
    if len(tags_list) ==1: #only one tag, show proportion and rank
        tag_rank = Table_tag_proportion(data,all_tags)
        rnk = tag_rank[tag_rank['tag'] == tags_list[0]].index.values[0]
        tags_cnt = len(all_tags)
        st.write(f'This tag ranks the {int(rnk)} among all {tags_cnt} games.')

def display_metric(metric_description:str,metric:str):
    st.metric(label=metric_description, value=metric)
    # st.caption(metric_description) 


def plot_mean_playtime_by_owners(data):
    # 创建Altair图表
    data = data[data['median_playtime'] > 0 ] 
    sorted_games = data.sort_values(by='median_playtime', ascending=False)
    sorted_games_display = sorted_games.head(min(sorted_games.shape[0],30))
    game_names = sorted_games_display['name'].tolist()
    
    color_scheme = alt.Scale(
        domain=['100M - 200M','50M - 100M', '20M - 50M', '10M - 20M', '5M - 10M', '2M - 5M','1M - 2M','0M - 1M',  '20W - 50W', '10W - 20W','5W - 10W','2W - 5W','0W - 2W'],
        range=['#00403d','#084c4a','#105957','#186664','#207370','#287f7d','#308c8a','#389997','#40a6a3','#48b2b0','#50bfbd','#58ccca','#60d9d6']  # 为 Owner1 和 Owner2 分别指定颜色
    )
    try:
        chart = alt.Chart(sorted_games_display).mark_bar().encode(
            x=alt.X('name:N', 
                    sort=game_names,
                    axis=alt.Axis(
                    labelAngle=-35,
                    labelOverlap=False,  # 强制显示所有标签
                    labelFontSize=12   # 旋转标签，负值表示逆时针旋转
                )),
            y='median_playtime',
            color=alt.Color('owners:N',scale=color_scheme),
            tooltip=['name', 'median_playtime', 'owners','developer']
        ).properties(
            title='Mean Playtime by Game and Owners',
            width=800,
            height=500
        )
        st.altair_chart(chart, use_container_width=False)
    except:
        st.write('Data missing for median playtime')

def plot_developer_barchart(data):
    # 创建Altair图表
    grouped_data = pd.DataFrame(data.groupby('developer').size()).reset_index()
    grouped_data.columns= ['developer','count']
    sorted_data = grouped_data.sort_values(by='count', ascending=False).head(min(grouped_data.shape[0],30))
    sorted_developers = sorted_data['developer'].tolist()
    sorted_counts = sorted_data['count'].tolist()
    chart = alt.Chart(sorted_data).mark_bar().encode(
        x=alt.X('developer:N', 
                sort=sorted_developers,
                axis=alt.Axis(
                labelAngle=-35,
                labelOverlap=False,  # 强制显示所有标签
                labelFontSize=12   # 旋转标签，负值表示逆时针旋转
            )),
        y= 'count',
        # color=alt.Color('owners:N',scale=color_scheme),
        tooltip=['developer', 'count']
    ).properties(
        title='Game count from developers',
        width=800,
        height=500
    )
    st.altair_chart(chart, use_container_width=True)


def plot_distribution(data, column, color='skyblue', highlight_value=None, highlight_color='red'):
    fig, ax = plt.subplots(figsize=(10, 4))
    # 绘制分布密度
    sns.kdeplot(data[column], fill=True, color=color, alpha=0.5, ax=ax,cut = 0)

    # 突出显示特定值
    if highlight_value is not None:
        # 在指定值处绘制垂直线
        ax.axvline(x=highlight_value, color=highlight_color, linestyle='--', 
                   label=f'Highlight: {highlight_value}')
        ax.scatter(highlight_value, 0, color=highlight_color, s=100, zorder=5)  # 高亮显示的点
        ax.set_xlim(-100, highlight_value + 5000)

    # 添加标题和标签
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

    # 显示图表
    st.pyplot(fig)


def plot_count_trend(tag_slice):
    tag_slice['release_date'] = pd.to_datetime(tag_slice['release_date'], errors='coerce')
    tag_slice['year'] = tag_slice['release_date'].apply(lambda x : x.year)
    tag_slice['year'] = tag_slice['year'].astype(int)
    summary = tag_slice.groupby('year').agg(game_count=('year', 'size')).reset_index()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=summary['year'], y=summary['game_count'], name="Game Count")
    )

    fig.update_xaxes(title_text="Year")
    fig.update_layout(
        title_text="Game Releases Over Years",
        hovermode="x unified",
        xaxis = dict(
            tickmode = 'linear',
            tick0 = min(summary['year']),  # 使用数据中最小的年份作为起始点
            dtick = 1  # 每个刻度间隔为1年
    ),
        yaxis = dict(range=[0, max(summary['game_count']) * 1.1],
                    tickmode = 'linear',
                    tick0 = 0,
                    dtick = int(max(summary['game_count'])/8)))
    # 在Streamlit中显示图表
    st.plotly_chart(fig, use_container_width=True)

def plot_time_trend(tag_slice,col_name,var_name):
    tag_slice['release_date'] = pd.to_datetime(tag_slice['release_date'], errors='coerce')
    tag_slice['year'] = tag_slice['release_date'].apply(lambda x : x.year)
    tag_slice['year'] = tag_slice['year'].astype(int)

    filtered_df = tag_slice[tag_slice[col_name] >= 0] 
    summary = filtered_df.groupby('year').agg(var_name=(col_name, 'mean')).reset_index()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=summary['year'], y=summary['var_name'], name= var_name)
)
    fig.update_xaxes(title_text="Year")
    title_name = var_name.replace('_',' ').capitalize()

    fig.update_layout(
        title_text=f"{title_name} Over Years",
        hovermode="x unified",
        xaxis = dict(tick0 = min(summary['year']),
                    tickmode = 'linear',  # 使用数据中最小的年份作为起始点
                    dtick = 1 ),
        yaxis = dict(#tick0 = 0,
                    range=[0, max(summary['var_name']) * 1.1],
                    dtick = int(max(summary['var_name']) * 1.1/5),  # Y轴范围从0开始，最大值稍微多出10%
                    title=var_name,  # Y轴标题
                    tickmode='linear',
                    # dtick = int(max(summary['var_name'])/5)) # 使用数据中最小的年份作为起始点),            
        ))
    st.plotly_chart(fig, use_container_width=True)

