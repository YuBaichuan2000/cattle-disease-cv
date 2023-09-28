import timm
import torch
import torchvision
from torchvision.transforms import transforms

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
            pretrained: bool = True,
            num_classes: int = 3,
    ):

        self.test_transform = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        if path_to_pretrained_model:
            self.model = timm.create_model(
                model_name,
                num_classes=num_classes)

            if path_to_pretrained_model.startswith("https://"):
                import requests
                from io import BytesIO
                response = requests.get(path_to_pretrained_model)
                checkpoint = torch.load(BytesIO(response.content), map_location=self.device)['state_dict']
            else:
                checkpoint = torch.load(path_to_pretrained_model, map_location=self.device)['state_dict']


            for key in list(checkpoint.keys()):
                if key.startswith('net.model.'):
                    checkpoint[key[len("net.model."):]] = checkpoint[key]
                    del checkpoint[key]

            self.model.load_state_dict(checkpoint)

            self.model.eval()
            # self.model = torch.load(
            #     path_to_pretrained_model, map_location=self.device
            # )
        else:
            Exception("No path to pretrained model provided")

        pass



    def predict(self, x):

        x = self.test_transform(x)
        # expand batch dimension
        x = x.unsqueeze(0)

        output_tensor = self.model(x)

        prob_tensor = torch.nn.Softmax(dim=1)(output_tensor)

        k = 5 if prob_tensor.shape[1] > 5 else prob_tensor.shape[1]

        top_k = torch.topk(prob_tensor, k=k, dim=1)

        probabilites = top_k.values.detach().cpu().numpy().flatten()

        indices = top_k.indices.detach().cpu().numpy().flatten()
        formatted_predictions = []

        for pred_prob, pred_idx in zip(probabilites, indices):
            predicted_label = index_to_class_labels[pred_idx].title()
            predicted_perc = pred_prob * 100
            formatted_predictions.append(
                (predicted_label, f"{predicted_perc:.3f}%"))

        return formatted_predictions

        pass
