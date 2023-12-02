from transformers import pipeline
import openai

# openai.api_key = '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'

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

import requests

import streamlit as st

key="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"

def img2text(url):
    # Use a pipeline as a high-level helper
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    text=pipe(url)[0]["generated_text"]

    return text

def generate(scenerio):
    template="""
    You are a STory teller:
    You can generate a short story based on a simple narrative, the story should be no more than 2o words;

    CONTEXT:{scenerio}
    STORY:
"""
    prompt=PromptTemplate(template=template,input_variables=["scenerio"])

    story_llm=LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",temperature=1), prompt=prompt,verbose=True)
    story=story_llm.predict(scenerio=scenerio)
    return story


# print(generate(img2text("a.jpg")))

def audio(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {key}"}

    payloads={"inputs":message}
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open("audio.flac","wb") as file:
        file.write(response.content)

# audio(generate(img2text("a.jpg")))

def main():

    st.set_page_config(page_title="Image to Audio Story")
    st.header("Turn image into audio story")
    uploaded_file=st.file_uploader("Choose an image...",type="jpg")
    if uploaded_file is not None:
        bytes_data=uploaded_file.getvalue()
        with open(uploaded_file.name,"wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file,use_column_width=True)
        scenerio=img2text(uploaded_file.name)
        story=generate(scenerio)
        audio(story)

        with st.expander("scenerio"):
            st.write(scenerio)
        with st.expander("story"):
            st.write("story")
        st.audio("audio.flac")

if __name__=="__main__":
    main()




