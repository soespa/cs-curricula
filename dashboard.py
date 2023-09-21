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

st.markdown('# Topic Model für die Kernlehrpläne Informatik Sekundarstufe I & II')

documents = pd.read_json(os.path.join('data', 'documents.json'))
sentences = pd.read_json(os.path.join('data', 'sentences.json'))

docs = sentences['sentence']


st.markdown('## Datengrundlage')

n_states = len(documents['bundesland'].unique())
n_curricula = len(documents.groupby(['bundesland', 'stufe']))

cols = st.columns(3)

with cols[0]:
    st.metric(label='Bundesländer', value=n_states)

with cols[1]:
    st.metric(label='Kernlehrpläne', value=n_curricula)

with cols[2]:
    st.metric(label='Sätze', value=len(sentences))

st.markdown('## Themen')

topic_model = BERTopic.load(os.path.join('model', 'topic_model_merged.pkl'))
#topic_model = BERTopic.load(os.path.join('model', 'topic_model_merged'))

df_topics = topic_model.get_topic_info().set_index('Topic')[['CustomName', 'Count', 'Representation']]

other_label = 'Sonstiges'

other = df_topics.loc[-1]
other.loc['CustomName'] = other_label

df_topics = df_topics.drop(-1)

df_topics.loc[-1] = other

df_topics = df_topics.rename({
    'CustomName': 'Label',
    'Representation': 'Schlüsselwörter',
    'Count': 'Anzahl Sätze'
}, axis=1)

label = df_topics['Label']

st.dataframe(df_topics, use_container_width=True)

st.download_button(
    label='Herunterladen',
    data=df_topics.to_json().encode('utf-8'),
    file_name='topics.json',
    mime='application/json'
)

probabilities = topic_model.probabilities_
topics = topic_model.get_topic_info()
labels = topics['CustomName'].tolist()[1:]
prop = pd.DataFrame(probabilities, columns=labels)
prop[OTHER_LABEL] = 1 - prop.sum(axis=1)



st.markdown('## Anteil der Themen insgesamt')

fig = go.Figure(
    data=[
        go.Pie(
            labels=label,
            values=prop.mean()
        )
    ],
)

st.plotly_chart(fig, use_container_width=True)


df = sentences.merge(documents, left_on='document', right_index=True, how='left')

st.markdown('## Schwerpunkt nach Stufe')

fig = plot_stufe(topic_model, df)

st.plotly_chart(fig, use_container_width=True)

st.markdown('Themenschwerpunkt nach Bundesland')

fig = plot_states(topic_model, df)

st.plotly_chart(fig, use_container_width=True)


fig = topic_model.visualize_documents(docs, custom_labels=True, hide_annotations=True)

fig.update_layout(
    template="plotly"
)

st.plotly_chart(fig, use_container_width=True)
