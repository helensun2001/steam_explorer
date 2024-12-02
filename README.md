import streamlit as st
from src.nlp import load_wordvec

st.set_page_config(page_title="Steam Game Explorer", page_icon="🎮")

st.write("# Welcome to the Steam Game Explorer! 🎮")

<!-- st.sidebar.success("Select a page above.") -->

st.markdown(
    """
    This application helps you discover new games based on your preferences and provides insights into our game database.
    
    ### What you can do:
    - Get personalized game recommendations
    - Explore visualizations of our game database
    
    To get started, select a page from the sidebar on the left.
    """
)
