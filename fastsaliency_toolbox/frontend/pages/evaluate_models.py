# Load important libraries 
import pandas as pd
import streamlit as st 
from PIL import Image
import numpy as np
import pandas as pd

@st.cache
def compute_sali(model_name, img):
    img = np.asarray(img, np.float32)
    return st.session_state.interface.run(model_name, img)

@st.cache
def compute_metrics(model, original_sal, sal):
    return st.session_state.interface.test(model, original_sal, sal)

def app():
    """This application shows how can our toolbox be used to evaluate how close the knowledge-distillation model is to the original model.
    """
    st.markdown("## Evaluate Knowledge-Distillation Process")

    # Upload the dataset and save as csv
    st.markdown("### But how close are the knowledge distilled models to the original model?") 
    st.write("\n")

    st.write(
        """
        We used for knowledge distillation the SALICON dataset. However, it might be beneficial to check how your model perform in your domain. Let's use our toolbox for this. In the following please upload the original image.
        """
    )
    models = ["AIM", "IMSIG", "SUN", "RARE2012", "BMS", "IKN", "GBVS", "SAM", "DGII", "UniSal"]
    path = "frontend/Images/"


    # Code to read a single file 
    uploaded_kd_image = st.file_uploader("Choose an image to test KD", type = ['jpg', 'png'])

    if uploaded_kd_image :
        original_image = Image.open(uploaded_kd_image)
        col0, col1, col2 = st.columns([7, 5, 7])
        col1.markdown("### Original Image")
        col1.image(original_image, width = 300)
        for i, name in enumerate(models):
            orig_model = st.file_uploader("Choose the image of the original model (" + models[i] + ")", type = ['jpg', 'png'])
            
            col0, col1, col2, col3, col4 = st.columns([2, 6, 6, 6, 2])
            col1.markdown("### " + models[i])

            col1.image(compute_sali(models[i], original_image), width = 300, clamp=True) 
            col2.markdown("### Original " + models[i])
            if orig_model:
                col2.image(orig_model, width = 300, clamp=True) 
                orig_model_image = Image.open(orig_model)
                my_metrics = compute_metrics(name, compute_sali(models[i], original_image), np.asarray(orig_model_image, dtype=np.float32))
                df = pd.DataFrame([my_metrics], columns=["MODEL", "NSS", "CC", "SIM"]) 
                col3.markdown("### Evaluation Score")
                col3.write(df)