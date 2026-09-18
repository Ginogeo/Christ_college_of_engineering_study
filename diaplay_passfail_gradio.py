import gradio as gr
import joblib
import pandas as pd
import os

# Load model
model = joblib.load("kmeans_model.pkl")


def predict_cluster(annual_income, spending_score):
    input_data = pd.DataFrame({
        "Income": [annual_income],
        "Spending_Score": [spending_score]
    })

    cluster = model.predict(input_data)[0]
    
    # Get distance to each cluster center
    distances = model.transform(input_data)[0]
    
    return f"Customer Segment (Cluster): {cluster}\nDistances to centers: {distances.round(2)}"


demo = gr.Interface(
    fn=predict_cluster,
    inputs=[
        gr.Number(label="Income", minimum=0, maximum=200, value=50),
        gr.Number(label="Spending_Score", minimum=1, maximum=100, value=50)
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Customer Segmentation (K-Means)",
    description="Predict customer cluster based on Annual Income and Spending Score."
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
