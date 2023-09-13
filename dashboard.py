import streamlit as st
import pandas as pd
from bertopic import BERTopic
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Topic Model",
    layout='wide'
)

st.markdown('# Topic Model für die Kernlehrpläne Informatik Sekundarstufe I & II')

documents = pd.read_json(os.path.join('data', 'documents.json'))
sentences = pd.read_json(os.path.join('data', 'sentences.json'))

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

df = topic_model.get_topic_info().set_index('Topic')[['CustomName', 'Count', 'Representation']]

other_label = 'Sonstiges'

df = df.drop(-1)

df = df.rename({
    'CustomName': 'Label',
    'Representation': 'Schlüsselwörter',
    'Count': 'Anzahl Sätze'
}, axis=1)

st.dataframe(df, use_container_width=True)

st.download_button(
    label='Herunterladen',
    data=df.to_json().encode('utf-8'),
    file_name='topics.json',
    mime='application/json'
)

probabilities = topic_model.probabilities_

prop = pd.DataFrame(probabilities, columns=df['Label']).mean()


st.markdown('## Anteil der Themen insgesamt')

fig = go.Figure(
    data=[
        go.Pie(
            labels=df['Label'],
            values=prop
        )
    ],
)


st.plotly_chart(fig, use_container_width=True)

