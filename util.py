import pandas as pd
from bertopic import BERTopic
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import pandas as pd
from bertopic import BERTopic
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

color_sek1 = 'rgb(180, 120, 20)'
color_sek2 = 'rgb(20, 120, 180)'


class CurriculaAnalysis():

    def __init__(self):
        self.documents = pd.read_json(os.path.join('data', 'documents.json'))
        self.sentences = pd.read_json(os.path.join('data', 'sentences.json'))

        self.docs = self.sentences['sentence']

        self.topic_model = BERTopic.load(os.path.join('model', 'topic_model_merged.pkl'))

        self.df = self.sentences.merge(self.documents, left_on='document', right_index=True, how='left')

        #self.nlp = stanza.Pipeline(lang='de', processors='tokenize,mwt,pos,lemma')

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

    def sentences_per_curriculum(self):
        lengths = self.df.groupby(['bundesland', 'stufe']).apply(lambda g: len(g)).unstack(level=1)


    def plot_level(self):
        df_props = self.df_props

        df_props = df_props[df_props['bundesland'].isin(COMPLETE_STATES)]

        df_curricula = df_props.groupby(['bundesland', 'stufe'], as_index=False).mean()

        df_level = df_curricula.drop('bundesland', axis=1).groupby('stufe').mean().T

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
                color = color_sek1
                text = f"+{row['diff'].round(1)}%"
            elif value < 0:
                color = color_sek2
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
                        size=11,
                        color=color
                    ),
                    showarrow=False
                )
            )

        return fig


    def plot_states(self):
        df_props = self.df_props

        df_props = (df_props.groupby(['bundesland', 'stufe']).mean() * 100)

        sek1 = df_props.xs('Sekundarstufe I', level=1).drop(OTHER_LABEL, axis=1)
        sek2 = df_props.xs('Sekundarstufe II', level=1).drop(OTHER_LABEL, axis=1)

        #sek1 = (sek1 - sek1.mean()) / sek1.std()

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                'Sekundarstufe I',
                'Sekundarstufe II'
            ],
            shared_xaxes=True,
            vertical_spacing=0.1
        )


        fig.add_trace(
            go.Heatmap(
                z=sek1,
                x=sek1.columns,
                y=sek1.index,
                text=sek1.round(1),
                texttemplate="%{text}",
                textfont={"size": 12},
                colorbar=dict(
                    y=0.25,
                    yanchor='middle',
                    lenmode="fraction",  # Length of the colorbar relative to the plot (fraction)
                    len=0.4,  # Set the length of the colorbar to span the entire plots
                )
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Heatmap(
                z=sek2,
                x=sek2.columns,
                y=sek2.index,
                text=sek2.round(1),
                texttemplate="%{text}",
                textfont={"size": 12},
                colorbar=dict(
                    y=0.75,
                    yanchor='middle',
                    lenmode="fraction",  # Length of the colorbar relative to the plot (fraction)
                    len=0.4,  # Set the length of the colorbar to span the entire plots
                )
            ),
            row=2,
            col=1
        )

        fig.update_yaxes(
            scaleanchor='x',
            scaleratio=1,
            #showgrid=False
        )

        fig.update_layout(
            #height=1200,
        )

        return fig


    def plot_topic_similarity(self):
        fig = self.topic_model.visualize_hierarchy(custom_labels=True)

        fig.update_layout(
            template="plotly",
            title='',
            xaxis=dict(
                title='Cosine Distance'
            )
        )

        return fig


    def plot_sentences(self):
        fig = self.topic_model.visualize_documents(self.docs, custom_labels=True, hide_annotations=True)

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


    def get_topic(self, topic_name, threshold=0.8):

        df_topics = self.df_topics

        topic = df_topics[df_topics['Thema'] == topic_name].index[0]

        df_props = self.df_props

        docs = self.df[df_props[topic_name] > threshold]

        docs = docs[['raw_sentence', 'bundesland', 'stufe', 'titel']]

        docs = docs.rename({
            'raw_sentence': 'Phrase',
            'bundesland': 'Bundesland',
            'stufe': 'Stufe',
            'titel': 'Abschnitt'
        }, axis=1)

        return docs

    def search_term(self, search_term, threshold=0.5):

        search_term = search_term.lower()

        topics, props = self.topic_model.find_topics(search_term=search_term)
        s = pd.Series(props, index=topics)

        s = s[s > threshold]

        result = None

        if len(s) > 0:
            df_topics = self.df_topics

            result = df_topics.loc[s.index]


        return result
