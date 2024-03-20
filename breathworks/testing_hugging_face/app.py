from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def classify(text):
    model_name = "bert-base-uncased"
    classifier = pipeline("text-classification", model=model_name)
    return classifier(text)


def generate_story(scenario):
    template = """
    you are a storyteller, you can generate a short story based on a simple narative
    , the story should be no more than 20 words. CONTEXT: {scenario}, story:"""

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)

    print(story)
    return story


def text2speech(speech):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": speech
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)



scenario = "a man and a women are looking out across a pier"
story = generate_story(scenario)
text2speech(story)


def main():
    st.set_page_config(page_title="Img 2 audio story", page_icon="🎧")
    st.header("Turn img into audio story")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        scenario = scenario
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

    st.audio("audio.flac")


if __name__ == "__main__":
    main()
