import numpy as np
import streamlit as st
import pandas as pd
from bertopic import BERTopic
import os
import plotly.graph_objects as go
from umap import UMAP
from .util import CurriculaAnalysis

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

st.markdown('''
Als Grundlage für die Analyse dienen 29 Kernlehrpläne für das Fach Informatik aus
16 Bundesländer für die Sekundarstufe I und II.
Zunächst wurden relevante Textpassagen in den Lehrplänen identifiziert und aus den PDFs extrahiert und gesammelt.
Anschließend wurden die Texte mit Hilfe von regulären Ausdrücken bereinigt.
Schließlich wurden die Texte in Phrasen zerlegt, welche die Grundlage für die Analyse darstellen.
''')

st.markdown('## Themen')

st.markdown('''
Für das Modellieren der Themen wurde [BERTopic](https://github.com/MaartenGr/BERTopic) verwendet.
Dabei handelt es sich um einen modulare Topic-Modelling Algorithmus.
Die einzelnen Schritte werden im folgenden kurz beschrieben:

1. Die Phrasen werden in einen hochdimensionalen Vektorraum eingebettet (Embeddings).
2. Anschließend wird eine Dimensionsreduzierungstechnik angewendet.
3. Die Vektoren werden mit Hilfe von Clustering-Algorithmen gruppiert. Jeder Cluster repräsentiert ein Thema.
4. Für jedes Thema werden die Schlüsselwörter (Wörter, die im Zusammenhang mit dem Thema häufig auftreten) ermittelt.

Im ersten Schritt mit Hilfe von BERTopic 66 Themen identifiziert, die anschließend manuell zu 20 Oberthemen gruppiert
wurden (siehe Tabelle), um die Übersichtlichkeit zu erhöhen.
Dabei sollte beachtet werden, dass eine solche Zusammenführung dazu führen kann, dass ein Thema Unterthemen enthält,
die nicht direkt aus dem Oberthema ersichtlich sind.

''')

df_topics = analysis.df_topics

st.dataframe(df_topics, use_container_width=True)

st.download_button(
    label='Herunterladen',
    data=df_topics.to_json().encode('utf-8'),
    file_name='topics.json',
    mime='application/json'
)

st.markdown('## Anteil der Themen insgesamt')

st.markdown('''
Im folgenden ist der Anteil der Themen im Durchschnitt über alle Lehrpläne dargestellt.
Als Grundlage für die Berechnung (und alle nachfolgenden Berechnungen) dient die Wahrscheinlichkeitsmatrix $A=(a_{ij})_{i=1,...,m;j=1,...,n}$, welche von dem Modell ermittelt wurde.
Der Eintrag $(a_{ij})$ ist die ermittelte Wahrscheinlichkeit, dass der i-te Satz zu dem j-ten Thema gehört.
''')

fig = analysis.get_total_topic_dist()

st.plotly_chart(fig, use_container_width=True)


with st.expander(label='Details'):
    st.markdown('## Ähnlichkeit der Themen untereinander')

    st.markdown('''
    Einige der Themen weisen größere Überschneidungen auf und können daher nicht immer präzise unterschieden werden.
    Bei der folgenden Darstellung handelt es sich um Dendrogram, welches die Ähnlichkeit der Themen untereinander zeigt.
    Umso ähnlicher sich zwei Themen sind, desto früher (weiter links) treffen sich die entsprechenden Zweige.
    ''')

    fig = analysis.plot_topic_similarity()

    st.plotly_chart(fig, use_container_width=True)


st.markdown('## Schwerpunkt nach Stufe')

st.markdown('''
Die nachfolgende Darstellung zeigt, wie sich der Schwerpunkt der Themen von der Sekundarstufe I zu der Sekundarstufe II
verschiebt.


''')

fig = analysis.plot_level()

st.plotly_chart(fig, use_container_width=True)

st.markdown('## Schwerpunkt nach Bundesland')

fig = analysis.plot_states()

st.plotly_chart(fig, use_container_width=True)


#fig = analysis.plot_sentences()

#st.plotly_chart(fig, use_container_width=True)

df_topics = analysis.df_topics

st.markdown('## Themen im Detail')

selection = st.selectbox(label='Thema', options=df_topics['Thema'])

threshold = st.slider(label='Grenzwert', value=0.8, min_value=0.5, max_value=1.0, step=0.05)

details = analysis.get_topic(selection, threshold)

st.dataframe(details)

st.markdown('## Schlüsselwortsuche')

search_term = st.text_input(label='Schlüsselwort', value='Künstliche Intelligenz')

result = analysis.search_term(search_term=search_term)

st.dataframe(result)