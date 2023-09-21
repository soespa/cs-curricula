import pandas as pd
from bertopic import BERTopic
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import pandas as pd
from bertopic import BERTopic
import os
import plotly.graph_objects as go
from umap import UMAP

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
    13: "Bewerten und Analysieren",
    14: "Variablen und Datentypen",
    15: "Strukturieren und Vernetzen",
    16: "Simulationen",
    17: "Künstliche Intelligenz und Maschinelles Lernen",
    18: "Roboter",
    19: "Suchmaschinen"
}


class CurriculaAnalysis():

    def __init__(self):
        self.documents = pd.read_json(os.path.join('data', 'documents.json'))
        self.sentences = pd.read_json(os.path.join('data', 'sentences.json'))

        self.docs = self.sentences['sentence']

        self.topic_model = BERTopic.load(os.path.join('model', 'topic_model_merged.pkl'))

        self.df = self.sentences.merge(self.documents, left_on='document', right_index=True, how='left')

    @property
    def df_topics(self):
        """
        :return:
        DataFrame for downloading
        """
        df_topics = self.topic_model.get_topic_info().set_index('Topic')[['CustomName', 'Count', 'Representation']]

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

    @property
    def df_props(self):
        probabilities = self.topic_model.probabilities_
        labels = self.topic_model.get_topic_info()['CustomName'].tolist()[1:]
        df_props = pd.DataFrame(probabilities, columns=labels)
        df_props[OTHER_LABEL] = 1 - df_props.sum(axis=1)

        df_props = pd.concat([self.df[['bundesland', 'stufe']], df_props], axis=1)

        return df_props

    def get_total_topic_dist(self):

        # Calculate mean for each curriculum
        prop = self.df_props.groupby(['bundesland', 'stufe']).mean()

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

    @property
    def n_states(self):
        return len(self.documents['bundesland'].unique())

    @property
    def n_curricula(self):
        return len(self.documents.groupby(['bundesland', 'stufe']))

    @property
    def n_sentences(self):
        return len(self.sentences)


    def plot_level(self):
        df_props = self.df_props

        df_props = df_props[df_props['bundesland'].isin(COMPLETE_STATES)]

        df_curricula = df_props.groupby(['bundesland', 'stufe'], as_index=False).mean()

        st.write(df_curricula)


        #df_level = df_curricula.groupby('bundesland', as_index=False).apply(lambda g: g.set_index('stufe').unstack())

        st.write(df_level)

        df_level['diff'] = (1 - (df_level['Sekundarstufe I'] / df_level['Sekundarstufe II'])) * 100

        df_level = df_level.sort_values('diff')

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
                )
            ],
            layout=go.Layout(
                height=800,
            )
        )

        for index, row in df_level.iterrows():
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

        return fig


def plot_states(topic_model: BERTopic, df: pd.DataFrame):
    topics = topic_model.get_topic_info()

    probabilities = topic_model.probabilities_
    label = topics['CustomName'].tolist()

    print(label)
    prop = pd.DataFrame(probabilities, columns=label[1:])

    prop[OTHER_LABEL] = 1 - prop.sum(axis=1)

    prop[df['bundesland'].isin(COMPLETE_STATES)]

    df_grouped = pd.concat([df[['bundesland', 'stufe']], pd.DataFrame(topic_model.probabilities_)], axis=1)
    df_agg = df_grouped.groupby(['bundesland', 'stufe']).mean()

    # agg = cur_df.rename(short_labels, axis=1)

    df_agg = df_agg.loc[(COMPLETE_STATES)]

    df_agg = df_agg.groupby('bundesland').mean()

    fig = go.Figure(
        data=go.Heatmap(
            z=df_agg,
            x=df_agg.columns,
            y=label,
            hoverongaps=False
        )
    )

    return fig
