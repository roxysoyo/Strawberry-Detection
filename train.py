import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch import nn
import math
from datasetup import StrawberryDataset
from engine import train
from model import VGG16
from utils import create_writer

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set data transform
data_transform = transforms.Compose(
    [
        transforms.Resize(size=(96, 96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)

# Give the file name and path
annotations_file = "Data/Annotations.csv"
img_dir = "Data/images"
dataset = StrawberryDataset(annotations_file, img_dir, transform=data_transform)

# Divide the training set and test set
train_ratio, test_ratio = 0.75, 0.25
train_dataset, test_dataset = random_split(
    dataset=dataset,
    lengths=[
        int(train_ratio * dataset.__len__()),
        math.ceil(test_ratio * dataset.__len__()),
    ],
)

# Load the dataset with Dataloader
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the model
model_0 = VGG16()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=LEARNING_RATE)

writer = create_writer()
# Train model_0
model_0_results = train(
    model=model_0,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    writer=writer,
)
writer.close()
print("ok")
