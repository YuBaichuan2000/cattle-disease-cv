import os

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from custom_model import CustomModel


@st.cache()
def load_model(path: str = 'deploy/cattle-disease-prod/models/epoch_003.ckpt') -> CustomModel:
    """Retrieves the trained model and maps it to the CPU by default,
    can also specify GPU here."""
    model = CustomModel(path_to_pretrained_model=path)
    return model


def load_list_of_images_available(image_files_subset, selected_species):
    base_path = os.path.join('deploy', 'cattle-disease-prod', 'imgs')

    if image_files_subset == 'All':
        images = []
        for split in ['train', 'valid', 'test']:
            species_path = os.path.join(base_path, split, selected_species)
            species_images = os.listdir(species_path)
            images += species_images
        return images

    return sorted(list(os.listdir(os.path.join(base_path, image_files_subset, selected_species))))

    pass


def read_image(image_files_subset, selected_species, image_name):
    base_path = os.path.join('deploy', 'cattle-disease-prod', 'imgs')

    image_file = os.path.join(base_path, image_files_subset, selected_species, image_name)

    image = Image.open(image_file)
    return image


def show_predict(selected_image, model):
    probs = model.predict(selected_image)
    # st.write(probs)
    st.title("Here is the image you've selected")
    resized_image = selected_image.resize((336, 336))
    st.image(resized_image)
    st.title("Here are classification scoes")
    df = pd.DataFrame(data=np.zeros((3, 2)),
                      columns=['Category', 'Confidence Level'],
                      index=np.linspace(1, 3, 3, dtype=int))
    for idx, p in enumerate(probs):
        df.iloc[idx,
        0] = p[0]  # category name
        df.iloc[idx, 1] = p[1]
    st.write(df.to_html(escape=False), unsafe_allow_html=True)


if __name__ == '__main__':

    import sys

    sys.path.append("../../")  # go to parent dir

    model = load_model()

    st.title('Welcome To Project Cattle Disease Classifications')
    instructions = """
        Either upload your own image or select from
        the sidebar to get a preconfigured image.
        The image you select or upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
    st.write(instructions)

    file = st.file_uploader('Upload An Image')
    dtype_file_structure_mapping = {
        # 'All Images': 'All',
        'Images The Model Has Never Seen': 'test',
        'Images Used To Train The Model': 'train',
        'Images Used To Tune The Model': 'valid',

    }
    data_split_names = list(dtype_file_structure_mapping.keys())

    cattle_type_mapping = {
        'NO Disease': 'Normal',
        'FMD': 'FMD',
        'LSD': 'LSD',
    }
    cattle_type_names = list(cattle_type_mapping.keys())

    dataset_type = st.sidebar.selectbox(
        "Data Portion Type", data_split_names)

    image_files_subset = dtype_file_structure_mapping[dataset_type]

    selected_species = st.sidebar.selectbox("Cattle Type", cattle_type_names)

    selected_species = cattle_type_mapping[selected_species]

    available_images = load_list_of_images_available(
        image_files_subset, selected_species)

    selected_image = st.sidebar.selectbox("Select Image", available_images)

    if file:

        selected_image = Image.open(file)
        # show_predict(selected_image,model)
        # file = False

    else:

        selected_image = read_image(image_files_subset, selected_species, selected_image)

    show_predict(selected_image,model)

        # img = Image.open(file)



    # st.title("Here is the image you've selected" + )
