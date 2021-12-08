import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Helper functions (Just for developing)!
def read_image(path, dtype=np.float32):
    f = Image.open(path)
    img = np.asarray(f, dtype)
    return img

@st.cache
def compute_sal(model_name, img, postprocessing_parameter_map, blur=0.0, hm=False):
    img = np.asarray(img, np.float32)
    return st.session_state.interface.run(model_name, img, postprocessing_parameter_map)

@st.cache
def compute_test(model, original_sal, sal):
    return st.session_state.interface.test(model, original_sal, sal)

# @st.cache
def app():
    st.markdown("## Compute Saliency")
    #st.session_state.interface.memory_check("TEST in COMPUTE")
    # Upload the dataset and save as csv
    st.markdown("### Upload an image to compute the saliency.") 
    st.write("\n")

    models = ["AIM", "IMSIG", "SUN", "RARE2012", "BMS", "IKN", "GBVS", "SAM", "DGII", "UniSal"]
    path = "frontend/Images/"

    #if 'computed_images' not in st.session_state:
	#    st.session_state.computed_images = [read_image(path + models[i].lower() + '.jpg') for i in range(10)]

    #if 'computed_state' not in st.session_state:
	#    st.session_state.computed_state = False 

    # Code to read a single file 
    uploaded_image = st.file_uploader("Choose an image", type = ['jpg', 'png'])

    ''' Compute the Saliency of the Images. '''
    if uploaded_image:
        original_image = Image.open(uploaded_image)

        col0, col1, col2 = st.columns([10, 5, 10])
        col1.markdown("### Original Image")
        col1.image(original_image, width = 300)

        blur = st.slider("You can also choose to blur the saliency images.", min_value=0, max_value=10, step=1)
        from backend.config import Config
        c = Config('config.json')
        if blur > 0.0:
            c.postprocessing_parameter_map.set("do_smoothing", "proportional")
            c.postprocessing_parameter_map.set("smooth_prop", blur/200)
        # settings
        hm = st.checkbox('Use histogram matching')
        if hm:
            c.postprocessing_parameter_map.set("histogram_matching", "biased")
        nrows, ncols = 2, 5  # array of sub-plots
        figsize = [7, 3]     # figure size, inches

        # create figure (fig), and array of axes (ax)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        my_bar = st.progress(0)
        # fig.delaxes(ax[3][2])
        # plot simple raster image on each sub-plot
        for i, axi in enumerate(ax.flat):
            axi.imshow(compute_sal(models[i], original_image, c.postprocessing_parameter_map, blur, hm), cmap='gray', vmin=0, vmax=1)
            axi.set_title(models[i],fontsize=6)
            axi.set_xticks([])
            axi.set_yticks([])
            my_bar.progress((i + 1)*10)

        st.pyplot(fig)
        st.markdown("## Do you have the ground truth? Let's evaluate our models!")

        # Upload the dataset and save as csv
        st.markdown("### Please upload the ground truth.") 
        st.write("\n")
        # Code to read a single file 
        uploaded_ground_truth = st.file_uploader("Choose the ground truth", type = ['jpg', 'png'])
        if uploaded_ground_truth and st.button("Evaluate Model"):
            ground_truth = Image.open(uploaded_ground_truth)
            col0, col1, col2 = st.columns([2, 5, 7])
            col1.write("Ground Truth")
            col1.image(ground_truth, width = 300)
            my_metrics = []
            print("Start Metric")
            for i, model in enumerate(models):
                print("Regetting the saliency")
                print("Trying to compute the metrics")
                my_values = compute_test(model, ground_truth, compute_sal(models[i], original_image, c.postprocessing_parameter_map, blur, hm))
                my_metrics.append(my_values)
                
            print(my_metrics)
            df = pd.DataFrame(my_metrics, columns=["Model", "NSS", "CC", "SIM"]) 
            col2.write("Evalutation Metrics")
            col2.write(df, width=300, height=100)





