import numpy as np
import pandas as pd
from bertopic import BERTopic
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

OTHER_LABEL = 'Sonstiges'

COMPLETE_STATES = [
    'Baden-Württemberg',
    'Bayern',
    'Berlin',
    'Brandenburg',
    'Hamburg',
    'Hessen',
    'Mecklenburg-Vorpommern',
    'Niedersachsen',
    'Nordrhein-Westfalen',
    'Rheinland-Pfalz',
    'Saarland',
    'Sachsen',
    'Schleswig Holstein'
]

custom_names = {
    0: "Informatiksysteme und Gesellschaft",
    1: "Informatische Sachverhalte kommunizieren und darstellen",
    2: "Netzwerkkommunikation und Verschlüsselung",
    3: "Daten und Informationen",
    4: "Objektorientierte Modellierung und Programmierung",
    5: "Formale Sprachen und Automaten",
    6: "Algorithmen und Datenstrukturen",
    7: "Softwareentwicklung und Projektmanagement",
    8: "Algorithmisches Problemlösen",
    9: "Datenbanken und SQL",
    10: "Berechenbarkeit und Komplexität",
    11: "Programmiersprachen, Funktionale und Logische Programmierung",
    12: "Fehlerbehebung und Debugging",
    13: "Analysieren und Bewerten",
    14: "Variablen und Datentypen",
    15: "Strukturieren und Vernetzen",
    16: "Simulationen",
    17: "Künstliche Intelligenz und Maschinelles Lernen",
    18: "Roboter",
    19: "Suchmaschinen"
}

color_sek1 = 'rgb(180, 120, 20)'
color_sek2 = 'rgb(20, 120, 180)'


@st.cache_resource
def load_data():
    documents = pd.read_json(os.path.join('data', 'documents.json'))
    sentences = pd.read_json(os.path.join('data', 'sentences.json'))

    docs = sentences['sentence']

    topic_model = BERTopic.load(os.path.join('model', 'topic_model_merged.pkl'))

    df = sentences.merge(documents, left_on='document', right_index=True, how='left')

    duality = pd.read_json(os.path.join('data', 'duality.json'))

    return documents, sentences, docs, topic_model, df, duality


documents, sentences, docs, topic_model, df, duality = load_data()

@st.cache_data
def get_states():
    return sorted(df['bundesland'].unique())

@st.cache_data
def get_level():
    return sorted(df['stufe'].unique())

@st.cache_resource
def get_df_topics():
    """
    :return:
    DataFrame for downloading
    """
    df_topics = topic_model.get_topic_info().set_index('Topic')[['CustomName', 'Count', 'Representation']]

    df_topics.index.name = None

    other = df_topics.loc[-1]
    other.loc['CustomName'] = OTHER_LABEL

    df_topics = df_topics.drop(-1)

    df_topics.loc[-1] = other

    df_topics = df_topics.rename({
        'CustomName': 'Thema',
        'Representation': 'Schlüsselwörter',
        'Count': 'Anzahl'
    }, axis=1)

    return df_topics


df_topics = get_df_topics()


@st.cache_resource
def get_df_props():
    probabilities = topic_model.probabilities_
    labels = topic_model.get_topic_info()['CustomName'].tolist()[1:]
    df_props = pd.DataFrame(probabilities, columns=labels)
    df_props[OTHER_LABEL] = 1 - df_props.sum(axis=1)

    df_props = pd.concat([df[['bundesland', 'stufe']], df_props], axis=1)

    return df_props


df_props = get_df_props()


@st.cache_resource
def get_total_topic_dist():

    # Calculate mean for each curriculum
    prop = df_props.groupby(['bundesland', 'stufe']).mean()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=prop.columns,
                values=prop.mean(),
                hole=0.3
            )
        ],
    )

    return fig

@st.cache_resource
def get_topic_dist_for_level():


    prop = df_props.groupby(['bundesland', 'stufe']).mean()

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=['Sekundarstufe I', 'Sekundarstufe II']
    )

    sek1 = prop.xs('Sekundarstufe I', level=1)
    sek2 = prop.xs('Sekundarstufe II', level=1)

    fig.add_trace(
        go.Pie(
            labels=sek1.columns,
            values=sek1.mean(),
            hole=0.3,
            showlegend=False,
            name='Sekundarstufe I'
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Pie(
            labels=sek2.columns,
            values=sek2.mean(),
            hole=0.3,
            showlegend=False,
            name='Sekundarstufe II'
        ),
        row=1,
        col=2
    )

    fig.update_layout(
        margin=dict(l=100, r=100, t=50, b=50),
    )

    return fig

@st.cache_data
def n_states():
    return len(documents['bundesland'].unique())


@st.cache_data
def n_curricula():
    return len(documents.groupby(['bundesland', 'stufe']))


@st.cache_data
def n_sentences():
    return len(sentences)

@st.cache_resource
def sentences_per_curriculum(self):
    lengths = self.df.groupby(['bundesland', 'stufe']).apply(lambda g: len(g)).unstack(level=1)

@st.cache_resource
def plot_level():

    props = df_props.copy()

    props = props[props['bundesland'].isin(COMPLETE_STATES)]

    df_curricula = props.groupby(['bundesland', 'stufe'], as_index=False).mean()

    df_level = df_curricula.drop('bundesland', axis=1).groupby('stufe').mean().T

    df_level = df_level*100

    df_level['diff'] = df_level['Sekundarstufe I'] - df_level['Sekundarstufe II']

    df_level = df_level.sort_values('diff')

    diff_trace = go.Bar(
                name='Differenz',
                y=df_level.index,
                x=df_level['diff'],
                orientation='h'
            )

    fig = go.Figure(
        data=[
            go.Bar(
                name='Sekundarstufe II',
                y=df_level.index,
                x=df_level['Sekundarstufe II'],
                orientation='h'
            ),
            go.Bar(
                name='Sekundarstufe I',
                y=df_level.index,
                x=df_level['Sekundarstufe I'],
                orientation='h'
            ),
            diff_trace
        ],
        layout=go.Layout(
            height=800,
            template='plotly_dark'
        )
    )

    for index, row in df_level.iterrows():
        value = row['diff']
        if value > 0:
            color = color_sek1
            text = f"+{row['diff'].round(2)}"
        elif value < 0:
            color = color_sek2
            text = f"{row['diff'].round(2)}"
        else:
            color = 'rgb(220, 220, 220)'
            text = f"{row['diff'].round(2)}"

        color = diff_trace.marker.color

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
                    size=11,
                    color=color
                ),
                showarrow=False
            )
        )

    return fig

@st.cache_resource
def plot_level_barpolar():

    props = df_props.copy()

    props = props[props['bundesland'].isin(COMPLETE_STATES)]

    props = props.drop(OTHER_LABEL, axis=1)

    df_curricula = props.groupby(['bundesland', 'stufe'], as_index=False).mean()

    df_level = df_curricula.drop('bundesland', axis=1).groupby('stufe').mean().T
    order = df_level.mean(axis=1).sort_values(ascending=False).index

    df_level = df_level.loc[order]

    fig = go.Figure(
        data=[
            go.Barpolar(
                name='Sekundarstufe II',
                theta=df_level.index,
                r=np.sqrt(df_level['Sekundarstufe II']),
                #fill='toself',
                base=0,
                opacity=0.8
            ),
            go.Barpolar(
                name='Sekundarstufe I',
                theta=df_level.index,
                r=np.sqrt(df_level['Sekundarstufe I']),
                #fill='toself',
                base=0,
                opacity=0.8
            )
        ],
        layout=go.Layout(
            #width=300,
            template='plotly_dark',
            barmode='overlay',
            margin=dict(l=200, r=200, t=50, b=50),
        )
    )


    return fig

@st.cache_resource
def plot_states(level=None):
    props = df_props.copy()

    props = (props.groupby(['bundesland', 'stufe']).mean() * 100).drop(OTHER_LABEL, axis=1)

    if level in ['Sekundarstufe I', 'Sekundarstufe II']:
        #df = props[props['stufe'] == level].drop('bunde')
        df = props.xs(level, level=1)
    elif level == 'Sekundarstufe I & II':

        df = props.reset_index()

        df = df[df['bundesland'].isin(COMPLETE_STATES)]

        df = df.drop('stufe', axis=1).groupby('bundesland').mean()


        #df = props[props['bundesland'].isin(COMPLETE_STATES)].groupby('bundesland').mean()

    fig = go.Figure(
        go.Heatmap(
            z=df,
            x=df.columns,
            y=df.index,
            text=df.round(1),
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )

    fig.update_layout(
        height=800,
        xaxis=dict(
            tickangle=60  # Set the desired tickangle (e.g., 45 degrees)
        )
    )

    return fig

@st.cache_resource
def plot_topic_similarity():
    fig = topic_model.visualize_hierarchy(custom_labels=True)

    fig.update_layout(
        template="plotly_dark",
        title='',
        xaxis=dict(
            title='Cosine Distance'
        )
    )

    return fig

@st.cache_resource
def plot_sentences():
    fig = topic_model.visualize_documents(docs, custom_labels=True, hide_annotations=True)

    fig.update_layout(
        template="plotly",
        legend=dict(
            yanchor="top",
            y=-0.01,
            xanchor="center",
            x=0.5
        ),
        height=1400
    )

    return fig

@st.cache_resource
def get_topic(topic_name, threshold=0.8):

    topic = df_topics[df_topics['Thema'] == topic_name].index[0]

    props = df_props

    docs = df[props[topic_name] > threshold]

    docs = docs[['raw_sentence', 'bundesland', 'stufe', 'titel']]

    docs = docs.rename({
        'raw_sentence': 'Phrase',
        'bundesland': 'Bundesland',
        'stufe': 'Stufe',
        'titel': 'Abschnitt'
    }, axis=1)

    docs = docs.reset_index(drop=True)

    return docs

@st.cache_resource
def search_for(search_term, threshold=0.5):

    search_term = search_term.lower()

    topics, props = topic_model.find_topics(search_term=search_term)


    s = pd.Series(props, index=topics)

    s = s[s > threshold]

    result = None

    if len(s) > 0:
        result = df_topics.loc[s.index]


    return result

@st.cache_resource
def get_curriculum(state, level):
    curriculum = df[(df['bundesland'] == state) & (df['stufe'] == level)]
    return curriculum

@st.cache_resource
def plot_duality():
    d = duality.copy()
    d = d.groupby(topic_model.topics_).mean()

    d = d.drop(-1)

    topics = topic_model.get_topic_info().set_index('Topic')

    d.index = map(lambda topic: topics.loc[topic]['CustomName'], d.index)

    order = (d['architecture'] / d['relevance']).sort_values().index

    d = d.loc[order]

    fig = go.Figure(
        data=[
            go.Bar(
                y=d.index,
                x=d['architecture'],
                name='Architektur',
                orientation='h'
            ),
            go.Bar(
                y=d.index,
                x=d['relevance'],
                name='Relevanz',
                orientation='h'
            )
        ],
        layout=go.Layout(
            height=40*len(topics),
            template='plotly_dark'
        )
    )

    return fig


