import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
import openai
import requests
import spacy
from spacy_streamlit import visualize_ner
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import plotly.express as px
from spacy_streamlit import visualize_parser
import nltk
nltk.download('punkt')
from nltk import tokenize




# Set Page Options
st.set_page_config(layout = 'wide')

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Advanced Natural Language Processing Insights and Analysis")

st.markdown("""
* Utilizes Several Transformer Architecture NLP Models (BERT, GPT3, GPT2, BART, etc)
* Programmed and Designed usign Streamlit Python API and Inference APIs
* Made by @sriramelango: https://github.com/sriramelango
""")

userMainOption = st.sidebar.selectbox(
    "Natural Language Processing Analytics Settings",
    ("Transformer Artificial Intelligence", "Text Analytics", "Dependence Structure")
)




# Functions for AI Requests
def openAIRequest(inputText, model, action, question = ""):
    if action == "Summary":
        inputText = "Summarize the text:\n'''''\n" + inputText + "'''''\nSummary:"
    elif action == "QA":
        inputText = inputText + "\n\n" + question
    response = openai.Completion.create(
        engine = model,
        prompt = inputText,
        temperature=0,
        max_tokens=250,
        top_p=1, 
        frequency_penalty=1,
        presence_penalty=1,
        stop=["'''''"]
    )
    return response.choices[0].text


def query(API_URL, payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


def linkToText(link):
    html = urlopen(link).read()
    soup = BeautifulSoup(html, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

# Functions for Analysis
def genWordCloud(text):
    stopwords = set(STOPWORDS)
    text = text.lower()
    wordCloud = WordCloud(background_color = "black", stopwords=stopwords, prefer_horizontal=1).generate(text) 
    plt.imshow(wordCloud, interpolation="bilinear") 
    plt.axis('off') 
    st.pyplot() 
        

def dataBarGraphProcess(data,xlabel,ylabel, unique):
    dataFiltered = []
    dataUnique = unique
    for i in range(len(dataUnique)):
        occurrences = data.count(dataUnique[i])
        dataFiltered.append([dataUnique[i], occurrences])
    dataFiltered = pd.DataFrame(dataFiltered, columns=[xlabel, ylabel])
    dataFiltered = dataFiltered.dropna()
    return dataFiltered
    

def plotSubjectivity(text):
    sentences = tokenize.sent_tokenize(text)
    data = []
    for i in range(len(sentences)):
        data.append(subClassification(sentences[i]))
    data = dataBarGraphProcess(data, "Perception", "Frequency", ["Objective", "Subjective"])
    fig = px.bar(data, x = "Perception", y= "Frequency", color = "Perception")
    st.plotly_chart(fig, use_container_width = True)


def subClassification(sentence):
    scoreSubjectivity = TextBlob(sentence).sentiment.subjectivity
    if scoreSubjectivity < 0.5:
        return "Objective"
    else:
        return "Subjective"


def plotSentiment(text):
    sentences = tokenize.sent_tokenize(text)
    data = []
    for i in range(len(sentences)):
        data.append(sentClassification(sentences[i]))
    data = dataBarGraphProcess(data, "Sentiment", "Frequency", ["Positive", "Negative", "Neutral"])
    fig = px.bar(data, x = "Sentiment", y= "Frequency", color = "Sentiment")
    st.plotly_chart(fig, use_container_width = True)


def sentClassification(sentence):
    sentimentScore = TextBlob(sentence).sentiment.polarity
    if sentimentScore < 0:
        return "Negative"
    elif sentimentScore == 0:
        return "Neutral"
    else:
        return "Positive"




# Main Application
if userMainOption == "Transformer Artificial Intelligence":

    GPT3TOKEN = st.text_input("GPT3 Token", 'Insert Token Here')
    openai.api_key = GPT3TOKEN
    HUGGINGFACETOKEN = st.text_input("Hugging Face Token", "Insert Token Here")
    headers = {"Authorization": "Bearer " + HUGGINGFACETOKEN}


    st.header("Text/Link Input")
    optionTextInput = st.selectbox("What input would you like to provide?", ("Text", "Link"))
    st.markdown("Text is Preferred!")


    if optionTextInput == "Text":
        text = st.text_area("Text/Corpus Input", "Input Text Here", height = 500)

    if optionTextInput == "Link":
        url = st.text_input("Website/Link Input", "Input Link Here")
        text = linkToText(url)
        

    # Abstract Summaries
    st.header("Abstract Summarization")
    optionSummaryModels = st.selectbox("What Transformer NLP Model?",
    ("GPT3 Davinci", "GPT3 Curie", "GPT2", "BART", "Distilbart", "Pegasus"))


    if optionSummaryModels == "GPT3 Davinci":

        if st.button("Summarize?"):

            with st.spinner('Wait for it...'):

                summary = openAIRequest(text, "text-davinci-002", "Summary")

            st.success('Done!')
            st.markdown(summary)



    if optionSummaryModels == "GPT3 Curie":

        if st.button("Summarize?"):

            with st.spinner('Wait for it...'):

                summary = openAIRequest(text, "text-curie-001", "Summary")

            st.success('Done!')
            st.markdown(summary)

        

    if optionSummaryModels == "Distilbart":

        API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-6-6"

        if st.button("Summarize?"):

            with st.spinner("Wait for it..."):

                summary = query(API_URL,{
                "inputs": text,
                })

            st.success("Done!")
            st.markdown(summary[0]["summary_text"])


    if optionSummaryModels == "BART":

        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

        if st.button("Summarize?"):

            with st.spinner("Wait for it..."):
                summary = query(API_URL,{
                "inputs": text,
                })

            st.success("Done!")
            st.markdown(summary[0]["summary_text"])


    if optionSummaryModels == "Pegasus":

            API_URL = "https://api-inference.huggingface.co/models/google/pegasus-xsum"

            if st.button("Summarize?"):

                with st.spinner("Wait for it..."):

                    summary = query(API_URL,{
                    "inputs": text,
                    })

                st.success("Done!")
                st.markdown(summary[0]["summary_text"])


    # Question and Answering
    st.header("Question and Answering")
    optionQAModels = st.selectbox("What NLP Model?", ("GPT3 Davinci", "GPT3 Curie","BERT-Large", "BERT-2", "Distilbert"))


    if optionQAModels == "GPT3 Davinci":

        st.info("More advanced questions can be asked here")
        question = st.text_input("What is your question?")

        if st.button("Question?"):

            with st.spinner('Wait for it...'):

                answer = openAIRequest(text, "text-davinci-002", "QA", question)

            st.success('Done!')
            st.markdown(answer)



    if optionQAModels == "GPT3 Curie":

        st.info("More advanced questions can be asked here")
        question = st.text_input("What is your question?")

        if st.button("Question?"):

            with st.spinner('Wait for it...'):

                answer = openAIRequest(text, "text-curie-001", "QA", question)

            st.success('Done!')
            st.markdown(answer)



    if optionQAModels == "BERT-Large":

        question = st.text_input("What is your question?")
        API_URL = "https://api-inference.huggingface.co/models/bert-large-cased-whole-word-masking-finetuned-squad"

        if st.button("Question?"):

            with st.spinner("Wait for it..."):

                answer = query(API_URL, {
                    "inputs": {
                        "question": question,
                        "context": text
                    },
                })  

            st.success("Done!")
            st.markdown("Confidence: " +  str(answer["score"]) + ", Answer: " + answer["answer"])
        

    if optionQAModels == "BERT-2":

        question = st.text_input("What is your question?")
        API_URL = "https://api-inference.huggingface.co/models/deepset/bert-large-uncased-whole-word-masking-squad2"

        if st.button("Question?"):

            with st.spinner("Wait for it..."):

                answer = query(API_URL, {
                    "inputs": {
                        "question": question,
                        "context": text
                    },
                })  

            st.success("Done!")
            st.markdown("Confidence: " +  str(answer["score"]) + ", Answer: " + answer["answer"])


    if optionQAModels == "Distilbert":

        question = st.text_input("What is your question?")
        API_URL = "https://api-inference.huggingface.co/models/distilbert-base-cased-distilled-squad"

        if st.button("Question?"):

            with st.spinner("Wait for it..."):

                answer = query(API_URL, {
                    "inputs": {
                        "question": question,
                        "context": text
                    },
                })  

            st.success("Done!")
            st.markdown(answer)



# Text Analysis           
if userMainOption == "Text Analytics":

    st.header("Text/Link Input")
    optionTextInput = st.selectbox("What input would you like to provide?", ("Text", "Link"))
    st.markdown("Text is Preferred!")

    if optionTextInput == "Text":
        text = st.text_area("Text/Corpus Input", "Input Text Here", height = 500)

    if optionTextInput == "Link":
        url = st.text_input("Website/Link Input", "Input Link Here")
        text = linkToText(url)

    # Entity Viewer
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    visualize_ner(doc, labels=nlp.get_pipe("ner").labels, show_table = False)

    # Entity Viewer
    st.header("Word Cloud")
    genWordCloud(text)

    # Perception Viewer
    st.header("Perception Data")
    plotSubjectivity(text)

    # Sentiment Viewer
    st.header("Sentiment Analysis")
    plotSentiment(text)


# Dependence Structure and Analysis        
if userMainOption == "Dependence Structure":

    st.header("Text/Link Input")
    optionTextInput = st.selectbox("What input would you like to provide?", ("Text", "Link"))
    st.markdown("Text is Preferred!")

    if optionTextInput == "Text":
        text = st.text_area("Text/Corpus Input", "Input Text Here", height = 500)

    if optionTextInput == "Link":
        url = st.text_input("Website/Link Input", "Input Link Here")
        text = linkToText(url)
    
    # Structure Viewer
    nlp = spacy.load("en_core_web_sm")
    text = nlp(text)
    visualize_parser(text)
