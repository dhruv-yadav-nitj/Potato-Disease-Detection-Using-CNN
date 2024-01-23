
import tensorflow as tf, numpy as np, streamlit as st, os
from PIL import Image
import requests
from io import BytesIO

# st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Potato Disease Detection')
st.text("Provide the URL for Input Image: ")

@st.cache(allow_output_mutation= True)
def load_model ():
    path = os.path.join(os.path.dirname(__file__), 'models')
    model = tf.keras.models.load_model(path)
    return model

model = None
with st.spinner('Loading Model Into Memory...'):
    model = load_model()

class_names = ['Early Blight', 'Late Blight', 'Healthy']

def scale(image):
    image= image.numpy().astype("uint8")
    image /= 255.0
    image= tf.image.resize(image, [256, 256])

def decode_img (image):
    img= tf.image.decode_jpeg(image, channels= 3)
    img= scale(img)
    return np.expand_dims(img, axis= 0)

path = st.text_input("Enter Image URL to Classify: ")

if path is not None:
    content = requests.get(path).content
    st.write("Prediction: ")
    with st.spinner("Classifying......"):
        label = np.argmax(model.predict(decode_img(content)), axis= 1)
        st.write(class_names[label[0]])
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption="Classifying Potato Disease", use_column_width=True)
