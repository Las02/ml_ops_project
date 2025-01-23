import time

import streamlit as st

input_text = None

st.write(
    "Welcome to our english to danish translater. Write your text in the box to translate text."
)
input_text = st.text_input("Translate from english to danish:")


_LOREM_IPSUM = """
Lorem ipsum dolor sit amet, **consectetur adipiscing** elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
"""


def stream_data():
    for word in _LOREM_IPSUM.split(" "):
        yield word + " "
        time.sleep(0.02)


if len(input_text) != 0:
    st.write("Translation:")
    st.write_stream(stream_data)
