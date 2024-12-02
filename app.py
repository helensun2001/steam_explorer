import streamlit as st

st.set_page_config(page_title="Steam Game Explorer", page_icon="🎮")

st.write("# Welcome to the Steam Game Explorer! 🎮")

#  st.sidebar.success("Select a page above.") 

st.markdown(
    """
    This application helps you explore historical games.
    
    ### What you can do:
    - Get personalized game recommendations
    - Explore genre analysis based on the dataset of over 20 thousand steam games.
    
    To get started, select Recommend page to get game recommendation based on freestyle input.
    Select Exlpore page to interact with the database and get insights.
    """
)
