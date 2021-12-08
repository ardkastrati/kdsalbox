import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports
from frontend.multipage import MultiPage
from frontend.pages import compute_saliency, task_evaluation, evaluate_models # import your pages here

st.set_page_config(layout="wide")

# Create an instance of the app
app = MultiPage()

@st.cache
def load_models():
    from backend.interface import Interface
    my_interface = Interface(gpu=-1)
    return my_interface

if 'interface' not in st.session_state:
	st.session_state.interface = load_models()

#st.session_state.interface.memory_check("TEST")

# Title of the main page
display = Image.open('frontend/Images/toolbox.png')
# st.title("Data Storyteller Application")
col0, col1, col2, col3 = st.columns([5,2,7,5])
col1.image(display, width = 100)
col2.title("Fast Saliency Toolbox")
# Add all your application here
app.add_page("Compute Saliency", compute_saliency.app)
app.add_page("Task Evaluation", task_evaluation.app)
app.add_page("Evaluate Knowldge-Distillation", evaluate_models.app)

# The main app
app.run()

st.sidebar.markdown(
    """
# Thank you!
Do you like this toolbox?

If you did, please give it a star on [GitHub](https://git.corp.adobe.com/adobe-research/fastsaliency-toolbox).

Created by Ard Kastrati and Zoya Bylinskii.
"""
)
