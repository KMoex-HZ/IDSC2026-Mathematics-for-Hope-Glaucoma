import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.models.model import GlaucomaEfficientNet
from src.data.dataloader import val_transform
from PIL import Image

def visualize_gradcam(img_path, model_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GlaucomaEfficientNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    target_layers = [model.backbone.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    rgb_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    rgb_img_float = np.float32(cv2.resize(rgb_img, (224, 224))) / 255.0

    input_tensor = val_transform(Image.fromarray(rgb_img)).unsqueeze(0).to(device)

    # FIX: Use targets=None for binary classification with a single output neuron.
    # ClassifierOutputTarget(0) is designed for multi-class (softmax) outputs and
    # is misleading here. With targets=None, GradCAM uses the raw scalar output
    # directly — which is correct for BCEWithLogitsLoss models.
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    cv2.imwrite("heatmap_result.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print("Grad-CAM visualization saved to heatmap_result.jpg")

if __name__ == '__main__':
    test_image = (
        "data/raw/hillel-yaffe-glaucoma-dataset-hygd-a-gold-standard-annotated-fundus-dataset"
        "-for-glaucoma-detection-1.0.0/Images/12_1.jpg"
    )
    visualize_gradcam(test_image)