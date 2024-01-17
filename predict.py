from utils import pred_and_plot_image
from model import VGG16
from torchvision import transforms
import torch


model = VGG16()
model.load_state_dict(torch.load("models/best.pth"))
class_names = ["ripe", "partripe", "unripe"]
image_path = "Data/prediction/v2.PNG"
image_size = (96, 96)

# Create transform pipleine to resize image
image_transform = transforms.Compose(
    [
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ]
)

pred_and_plot_image(model, class_names, image_path, image_size, image_transform)
