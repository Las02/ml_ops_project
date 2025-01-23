import time

import requests
import streamlit as st


def request_translate(input):
    # url = "https://docker-image-1-533116583496.europe-west1.run.app"
    url = "http://0.0.0.0:8000/"
    headers = {"accept": "application/json"}
    params = {"input": input}

    response = requests.post(url, headers=headers, params=params)

    return response.json()


def stream_data():
    for word in translation.split(" "):
        yield word + " "
        time.sleep(0.02)


st.write(
    "Welcome to our danish to english translater. Write your text in the box to translate text."
)
input_text = st.text_input("Translate from danish to english:")


if len(input_text) != 0:
    st.write("Translation:")
    translation = request_translate(input_text)
    st.write_stream(stream_data)
