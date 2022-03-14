# Import necessary libraries
import json
import pandas as pd
from pandas.core.arrays.sparse import dtype
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Helper functions (Just for developing)!
def read_image(path, dtype=np.float32):
    f = Image.open(path)
    img = np.asarray(f, dtype)
    return img

@st.cache
def compute_sal(model_name, img):
    img = np.asarray(img, np.float32)
    return st.session_state.interface.run(model_name, img)

@st.cache
def compute_task(model, original_sal, annotation):
    return st.session_state.interface.evaluate_task(model, original_sal, annotation)

def app():
    """This application shows how can our toolbox be used to evaluate which of the saliency models is more appropriate
    for a given task.
    """
    st.markdown("## Task Evaluation")

    # Upload the dataset and save as csv
    st.markdown("### You wonder which of the saliency models is more appropriate for your computer vision task?") 
    st.write("\n")

    st.write(
        """
        Let's use our toolbox for this. In the following please upload the original image and the image with annotations (in greyscale).
        """
    )

    models = ["AIM", "IMSIG", "SUN", "RARE2012", "BMS", "IKN", "GBVS", "SAM", "DGII", "UniSal"]
    path = "frontend/Images/"

    if 'computed_images' not in st.session_state:
	    st.session_state.computed_images = [Image.open(path + models[i].lower() + '.jpg') for i in range(10)]
        
    # Code to read a single file 
    uploaded_image = st.file_uploader("Choose the image", type = ['jpeg', 'jpg', 'png'])

    # Code to read a single file 
    uploaded_annotation = st.file_uploader("Choose the annotations", type = ['jpeg', 'jpg', 'png'])

    if uploaded_image and uploaded_annotation:
        original_image = Image.open(uploaded_image)
        annotation_image = Image.open(uploaded_annotation)
        annotation_image_np = np.asarray(annotation_image, dtype=np.float32)/255.0
        print(annotation_image_np)
        col0, col1, col2, col3 = st.columns([2, 5, 5, 2])
        col1.markdown("### Original Image")
        col1.image(original_image, width = 300)
        col2.markdown("### Annotation Image")
        col2.image(annotation_image, width = 300)

        # settings
        nrows, ncols = 2, 5  # array of sub-plots
        figsize = [7, 3]     # figure size, inches

        # create figure (fig), and array of axes (ax)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        my_bar = st.progress(0)
        # fig.delaxes(ax[3][2])
        # plot simple raster image on each sub-plot
        for i, axi in enumerate(ax.flat):
            axi.imshow(compute_sal(models[i], original_image), cmap='gray', vmin=0, vmax=1)
            axi.set_title(models[i],fontsize=6)
            axi.set_xticks([])
            axi.set_yticks([])
            my_bar.progress((i + 1)*10)

        st.pyplot(fig)

        if st.button("Evaluate Task"):
            col0, col1, col2 = st.columns([2, 5, 2])
            my_metrics = []
            print("Start Metric")
            my_bar2 = st.progress(0)
            for i, model in enumerate(models):
                my_values = compute_task(model, compute_sal(models[i], original_image), annotation_image_np)
                my_metrics.append(my_values)
                my_bar2.progress((i + 1)*10)

            df = pd.DataFrame(my_metrics, columns=["Model", "Precision", "Recall", "F1", "Accuracy"]) 
            col1.write("Evalutation Metrics for the Task")
            col1.write(df, width=300, height=100)

        

        




