import requests
import torch
import torchvision
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
import timm

app = Flask(__name__)

index_to_class_labels = {
    0: 'FMD',
    1: 'LSD',
    2: 'NO-Disease'
}


class CustomModel:
    def __init__(
            self,
            path_to_pretrained_model: str = "https://vmv.re/Cattle-Disease-Classification-Checkpoint-v1.pth",
            model_name: str = 'efficientnet_b0',
            num_classes: int = 3,
    ):
        # Compose transformations for the test data
        self.test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Define the device to be used
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Check the path to the pre-trained model
        if path_to_pretrained_model:
            self.model = timm.create_model(model_name, num_classes=num_classes)

            # If the model is located on the internet, download it
            if path_to_pretrained_model.startswith("https://"):
                response = requests.get(path_to_pretrained_model)
                checkpoint = torch.load(BytesIO(response.content), map_location=self.device)['state_dict']
            else:
                checkpoint = torch.load(path_to_pretrained_model, map_location=self.device)['state_dict']

            # Remove the prefix 'net.model.' from keys if present
            for key in list(checkpoint.keys()):
                if key.startswith('net.model.'):
                    checkpoint[key[len("net.model."):]] = checkpoint[key]
                    del checkpoint[key]

            # Load the state dict to the model
            self.model.load_state_dict(checkpoint)
            self.model.eval()
        else:
            raise Exception("No path to pretrained model provided")

    def predict(self, img):
        # Apply transformations, expand dimensions for batch and perform inference
        x = self.test_transform(img).unsqueeze(0)
        output_tensor = self.model(x)

        # Apply softmax to get probabilities
        prob_tensor = torch.nn.Softmax(dim=1)(output_tensor)

        # Get top 5 predictions or all if less than 5 classes
        k = min(5, prob_tensor.shape[1])
        top_k = torch.topk(prob_tensor, k=k, dim=1)
        probabilities = top_k.values.detach().cpu().numpy().flatten()
        indices = top_k.indices.detach().cpu().numpy().flatten()

        # Format predictions with class label and percentage
        formatted_predictions = [(index_to_class_labels[idx].title(), f"{prob * 100:.3f}%") for idx, prob in
                                 zip(indices, probabilities)]
        return formatted_predictions


# Load model
model = CustomModel()


@app.route('/predict', methods=['POST'])
def predict():
    # Check file in request
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read the image, convert it to RGB
    img = Image.open(BytesIO(file.read())).convert('RGB')

    # Make prediction
    predictions = model.predict(img)
    return jsonify(predictions)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
