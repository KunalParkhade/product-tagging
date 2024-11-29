import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from base_model import GarmentClassifier  # Model definition file

# Define class labels
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_model(model_path, device):
    """
    Loads the trained model from the specified path.
    """
    model = GarmentClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def predict(image, model, device, classes):
    """
    Predicts the class of a single image using the trained model.
    """
    # Preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale for Fashion MNIST
        transforms.Resize((28, 28)),  # Resize to match model input
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Forward pass
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)  # Compute probabilities
        _, predicted = torch.max(probabilities, 1)  # Get class index

    predicted_class = classes[predicted.item()]
    confidence = probabilities[0, predicted.item()] * 100  # Confidence score in percentage
    return predicted_class, confidence.item()

def main():
    """
    Streamlit app main function.
    """
    # Streamlit UI setup
    st.title("Product Tagging - Interactive Garment Classifier")
    st.write("Upload a garment image to predict its category.")

    # Sidebar navigation
    st.sidebar.header("Navigation")
    st.sidebar.markdown("Use this app to classify garments from the Fashion MNIST dataset.")

    # File uploader
    uploaded_file = st.file_uploader("Upload a garment image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model('./models/garment_classifier.pth', device)

        # Predict
        try:
            predicted_class, confidence = predict(image, model, device, classes)
            st.success(f"Predicted Class: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
