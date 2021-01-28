import sys
sys.path.append('../')
import streamlit as st
from tag_prediction_pipeline import TagPredictor
import pandas as pd


@st.cache(allow_output_mutation=True)
def load_predictor():
    tp = TagPredictor()
    return tp


if __name__ == '__main__':
    st.title('Tag for Movies')
    tp = load_predictor()

    col1, col2 = st.beta_columns(2)


    synopsis_input = col1.text_area('Plot Synopsis')
    # synopsis_file_uploader = st.file_uploader('Or, upload a file containing a synopsis')

    review_input = col2.text_area('Review')
    # review_file_uploader = st.file_uploader('Or, upload a file containing a review')

    # if synopsis_file_uploader:
    #     # stringio = StringIO(synopsis_file_uploader.decode("utf-8"))
    #     string_data = synopsis_file_uploader.read()
    #     # synopsis_input.value = string_data

    prediction = []
    if synopsis_input and review_input:
        prediction = tp.get_prediction(synopsis_input, review_input)
    elif synopsis_input:
        prediction = tp.get_prediction(synopsis_input, "")

    st.text(f"Top five tags: {', '.join(prediction)}")

