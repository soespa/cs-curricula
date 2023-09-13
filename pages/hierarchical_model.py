import streamlit as st
import pandas as pd
from bertopic import BERTopic
import os

st.markdown('# Hierarchical Model')

sentences = pd.read_json(os.path.join('data', 'sentences.json'))


topic_model = BERTopic.load(os.path.join('model', 'topic_model.pkl'))


hierarchical_topics = pd.read_json(os.path.join('data', 'hierarchical_topics.json'))


fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

st.plotly_chart(fig)