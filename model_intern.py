import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("abc.csv")


a = st.text_input("Please enter your reviews")
if a:
    a = [a]

    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(df['punctuation_removed'])
    a= count_vect.transform(a)

    loaded_model = pickle.load(open("xebia_internship_model.sav", 'rb'))
    predct = loaded_model.predict(a)

    if(predct[0] == 0):
        st.write("Negative")
    else:
        st.write("Positive")