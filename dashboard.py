import numpy as np
import streamlit as st
import pandas as pd
from bertopic import BERTopic
import os
import plotly.graph_objects as go
from umap import UMAP

from util import *

st.set_page_config(
    page_title="Topic Model",
    #layout='wide',
    initial_sidebar_state='collapsed'
)

analysis = CurriculaAnalysis()

st.markdown('# Topic Model für die Kernlehrpläne Informatik Sekundarstufe I & II')

st.markdown('## Datengrundlage')

cols = st.columns(3)

with cols[0]:
    st.metric(label='Bundesländer', value=analysis.n_states)

with cols[1]:
    st.metric(label='Kernlehrpläne', value=analysis.n_curricula)

with cols[2]:
    st.metric(label='Sätze', value=analysis.n_sentences)

st.markdown('## Themen')


df_topics = analysis.df_topics

st.dataframe(df_topics, use_container_width=True)

st.download_button(
    label='Herunterladen',
    data=df_topics.to_json().encode('utf-8'),
    file_name='topics.json',
    mime='application/json'
)

st.markdown('## Anteil der Themen insgesamt')

fig = analysis.get_total_topic_dist()

st.plotly_chart(fig, use_container_width=True)




st.markdown('## Schwerpunkt nach Stufe')

fig = analysis.plot_level()

st.plotly_chart(fig, use_container_width=True)

st.markdown('Themenschwerpunkt nach Bundesland')

fig = plot_states(topic_model, df)

st.plotly_chart(fig, use_container_width=True)


fig = topic_model.visualize_documents(docs, custom_labels=True, hide_annotations=True)

fig.update_layout(
    template="plotly"
)

st.plotly_chart(fig, use_container_width=True)
