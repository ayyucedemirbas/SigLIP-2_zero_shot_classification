import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "zero-shot-image-classification",
    model="google/siglip2-base-patch16-224",
    device=-1
)

def classify_image(image, candidate_labels):
    """
    Takes an image and a comma-separated string of candidate labels,
    and returns the classification scores.
    """
    labels = [label.strip() for label in candidate_labels.split(",") if label.strip()]

    results = classifier(image, candidate_labels=labels)
    return results[0]

iface = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(value="cat, dog, bird, car, airplane", label="Candidate Labels (comma separated)")
    ],
    outputs=gr.JSON(label="Classification Results"),
    title="SigLIP Zero-Shot Image Classifier",
    description="This app uses the Google SigLIP model (siglip2-base-patch16-224) for zero-shot image classification on CPU. "
                "Enter an image and a set of candidate labels to see the prediction scores."
)

if __name__ == "__main__":
    iface.launch()
