import streamlit as st
import requests
import os
from streamlit_navigation_bar import st_navbar

st.set_page_config(initial_sidebar_state="collapsed")

pages = ["Home", "About", "Tutorials", "Development"]
styles = {
    "nav": {
        "background-color": "rgb(123, 209, 146)",
    },
    "div": {
        "max-width": "50rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}

page = st_navbar(pages, styles=styles)
st.write(page)

with st.sidebar:
    st.write("Sidebar")

# Function to call the FastAPI predict endpoint
def call_predict_api(file):
    url = "http://127.0.0.1:8000/predict/"
    files = {'file': file}
    response = requests.post(url, files=files)
    return response.json()


def call_predict_api2(file):
    url = "http://127.0.0.1:8000/predict2/"
    files = {'file': file}
    response = requests.post(url, files=files)
    return response.json()


# Streamlit UI
st.title(':Gray[Glaucoma Detection Application]')
st.write("Upload an image to classify")
st.write("Impressive accuracy of 90%")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

response = None

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    # Display the uploaded image
    col1.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with st.spinner('SEGMENTING AND CLASSIFYING THE IMAGE...'):
        response = call_predict_api2(uploaded_file)

        if response:
            print(response)
            col2.markdown('<div style="text-align: center;"><h2>Result</h2></div>', unsafe_allow_html=True)
            col2.write(f"Filename: {response['filename']}")
            col2.write(f"score: {response['probability']}")
            if response['final verdict']:
                col2.write(f"Final Verdict: glaucomas")
            else:
                col2.write(f"Final Verdict: non-glaucomas")

col1, col2, col3 = st.columns(3)
if response is not None:
    col1.image(f"result/disc_seg_{response['filename']}", caption="disc segmentation Image", use_column_width=True)
    col2.image(f"result/cup_seg_{response['filename']}", caption="cup segmentation Image", use_column_width=True)
    col3.image(f"result/full_seg_{response['filename']}", caption="both segmentation Image", use_column_width=True)