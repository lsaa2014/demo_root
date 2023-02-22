import streamlit as st
import housePrice_st
import movie_rec
import adtk_st

import plotly.io as pio
from PIL import Image


pio.templates.default = "none"


img = 'ml.jpg'

## Put the logo image
st.sidebar.image(Image.open(img))

PAGES = {
 "House Prediction": housePrice_st,
 "Time Series": adtk_st,
 "NLP": movie_rec
}
st.sidebar.subheader("About")
st.sidebar.markdown("Real world examples of ML.")

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.write()

st.sidebar.markdown('''
    -------
    By Danielle Taneyo, PhD''')
