import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the model
model = tf.keras.models.load_model("animal_cnn.keras")

# 2. Define the prediction logic
def predict(image):
    if image is None:
        return "Please upload an image."
        
    # Format image for the model
    img = Image.fromarray(image).convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Return result based on threshold
    if prediction >= 0.5:
        return f"Prediction: Dog 🐶 (Confidence: {prediction*100:.1f}%)"
    else:
        return f"Prediction: Cat 🐱 (Confidence: {(1-prediction)*100:.1f}%)"

# 3. Create and launch the interface
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs="text",
    title="Simple Cat vs Dog Classifier"
)

if __name__ == "__main__":
    app.launch()