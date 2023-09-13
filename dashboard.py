import streamlit as st
import pandas as pd
from bertopic import BERTopic
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Topic Model",
    layout='wide',
    initial_sidebar_state='collapsed'
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

df_topics = topic_model.get_topic_info().set_index('Topic')[['CustomName', 'Count', 'Representation']]

other_label = 'Sonstiges'

df_topics = df_topics.drop(-1)

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

prop = pd.DataFrame(probabilities, columns=label)


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

df_stufe = prop.groupby(df['stufe']).mean().T

df_stufe['diff'] = (1 - (df_stufe['Sekundarstufe I'] / df_stufe['Sekundarstufe II'])) * 100

df_stufe = df_stufe.sort_values('diff')


fig = go.Figure(
    data=[
        go.Bar(
            name='Sekundarstufe II',
            y=df_stufe.index,
            x=df_stufe['Sekundarstufe II'],
            orientation='h'
        ),
        go.Bar(
            name='Sekundarstufe I',
            y=df_stufe.index,
            x=df_stufe['Sekundarstufe I'],
            orientation='h'
        )
    ],
    layout=go.Layout(
        height=800,
    )
)


for index, row in df_stufe.iterrows():
    value = row['diff']
    if value > 0:
        color = 'rgb(80, 180, 80)'
        text = f"+{row['diff'].round(1)}%"
    elif value < 0:
        color = 'rgb(180, 80, 80)'
        text = f"{row['diff'].round(1)}%"
    else:
        color = 'rgb(220, 220, 220)'
        text = f"{row['diff'].round(1)}%"

    fig.add_annotation(
        dict(
            xref='x',
            yref='y',
            x=row[['Sekundarstufe I', 'Sekundarstufe II']].max(),
            y=index,
            xanchor='left',
            text=text,
            font=dict(
                family='Arial',
                size=14,
                color=color
            ),
            showarrow=False
        )
    )

st.plotly_chart(fig, use_container_width=True)

