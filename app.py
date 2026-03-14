import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load trained model
model = tf.keras.models.load_model("models/road_damage_model.h5")

class_names = ['crack','manhole','pothole']

st.title("🚧 Road Damage Detection System")

uploaded_file = st.file_uploader("Upload a road image", type=["jpg","jpeg","png"])


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap,0) / np.max(heatmap)

    return heatmap


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224,224))

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("Prediction Result")

    st.write(f"### Detected Damage: {predicted_class}")
    st.write(f"### Confidence Score: {confidence:.2f}%")

    if predicted_class == "pothole":
        st.warning("⚠️ Recommended Action: Immediate road repair required.")

    elif predicted_class == "crack":
        st.warning("⚠️ Recommended Action: Monitor and schedule maintenance.")

    elif predicted_class == "manhole":
        st.info("ℹ️ Recommended Action: Check manhole cover alignment.")

    heatmap = make_gradcam_heatmap(img_array, model, "Conv_1")

    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(np.array(image),0.6,heatmap,0.4,0)

    st.subheader("Grad-CAM Visualization")

    st.image(superimposed_img, use_container_width=True)