# Import Libraries
import os
import json
import torch
import torchvision
from PIL import Image
import glob
import matplotlib.pyplot as plt
from random import shuffle

# Load Config files
path = '/content/drive/My Drive/YAAR/yaar_'
config_path = os.path.join(path, 'config.json')
with open(config_path, 'r') as f:
  config = json.load(f)

print("The Configuration Variables are:")
print(config)

# Define Config variables
image_size = config['image_size']
data_path = config['DataPath'] 
batch_size = config['batch_size']
learning_rate = config['lr']
weight_decay = config['weight_decay']
epochs = config['n_epochs']

print("\n____________________________________________________\n")
print("\nLoading Dataset into DataLoader...")

# Get All Images
Cat_Imgs = os.path.join(data_path, 'Cat/')
Cat_Imgs = glob.glob(Cat_Imgs + '*')

Dog_Imgs = os.path.join(data_path, 'Dog/')
Dog_Imgs = glob.glob(Dog_Imgs + '*')

all_imgs = Cat_Imgs + Dog_Imgs
shuffle(all_imgs)

# Train Images
train_imgs = all_imgs[:1950]
test_imgs = all_imgs[1950:]

# DataLoader Function
class imagePrep(torch.utils.data.Dataset):
    def __init__(self, images, transform):
        super().__init__()
        self.paths = images 
        self.len = len(self.paths)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path)
        image = self.transform(image)
        if 'Cat' in path:
            label = 0
        else:
            label = 1
        
        return (image, label)

# Dataset Transformation Function
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                        torchvision.transforms.Resize((image_size,image_size)),
                                        torchvision.transforms.ToTensor()])

# Apply Transformations to Data
Tr_DL = torch.utils.data.DataLoader(imagePrep(train_imgs, dataset_transform), batch_size= batch_size)
Ts_DL = torch.utils.data.DataLoader(imagePrep(test_imgs, dataset_transform), batch_size= batch_size)


# Open one image
print("\nTest Open One Image")
plt.imshow(Image.open(all_imgs[5]))

print("\nDataLoader Set!")
print("\n____________________________________________________\n")

print("\nBuilding Convolutional AutoEncoder Network Model...")

# Define Convolutional AutoEncoder Network
class ConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=1, padding=1),  # 
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1),
            torch.nn.Conv2d(64, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(16, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(64, 1, 3, stride=1, padding=2),  # b, 8, 3, 3
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)
        return decoded


print("\nConvolutional AutoEncoder Network Model Set!")

print("\n____________________________________________________\n")

# defining the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining the model
convAE_model = ConvAutoencoder().to(device)

# defining the optimizer
optimizer = torch.optim.Adam(convAE_model.parameters(), lr= learning_rate, weight_decay= weight_decay)

# defining the loss function
loss_function = torch.nn.MSELoss().to(device)

print(convAE_model)
print("____________________________________________________\n")

print("\nTraining the Convolutional AutoEncoder Model on Training Data...")

# Training of Model

losses = []
for epoch in range(epochs): 
    epoch_loss = 0
    for X, y in Tr_DL:
        img = X.to(device)
        img = torch.autograd.Variable(img)
    
        recon = convAE_model(img)

        loss = loss_function(recon, img)
        
        # Backward Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+= loss
        print('-', end= "", flush= True)

    epoch_loss = epoch_loss/len(Tr_DL)
    losses.append(epoch_loss)

    print("\nEpoch: {} | Loss: {:.4f}".format(epoch+1, epoch_loss))

print("\n____________________________________________________\n")

fig = plt.figure(figsize = (12,5))

plt.plot(losses, '-r', label='Training loss')
plt.xlabel('Epochs', fontsize= 15)
plt.ylabel('Loss', fontsize= 15)
plt.title('Convolutional AutoEncoder Training Loss Vs Epochs', fontsize= 15)
plt.show()

print("\n____________________________________________________\n")

print("PRINTING ORIGINAL IMAGES THAT TRAINED THE MODEL AND THEIR RECONSTRUCTIONS ...")

# Print Some Reconstructions
plt.figure(figsize = (23, 8))

start = 4
n_images = 5

for i in range(n_images):
    plt.subplot(1, n_images, i+1)
    plt.imshow(X[start+i+1][0], cmap='gray')
    plt.title('Training Image ' + str(i+1), fontsize = 15)
    plt.axis("off")

plt.figure(figsize = (23, 8))

for i in range(n_images):
    plt.subplot(1, n_images, i+1)
    pic = recon.cpu().data
    plt.imshow(pic[start+i+1][0], cmap='gray')
    plt.title('Reconstructed Image ' + str(i+1), fontsize = 15)
    plt.axis("off")

print("\n____________________________________________________\n")

print("\n____________________________________________________\n")

# Reconstruct Images by passing Test images on Trained Model
with torch.no_grad():
    for Ts_X, Ts_y in Ts_DL:
        Ts_X = Ts_X.to(device)
        Ts_y = Ts_y.to(device)

        Ts_recon = convAE_model(Ts_X)
        

print("PRINTING TEST IMAGES AND THEIR RECONSTRUCTIONS ...")
print("\n____________________________________________________\n")

# Print Some Reconstructions
plt.figure(figsize = (23, 8))

start = 4
n_images = 5

for i in range(n_images):
    plt.subplot(1, n_images, i+1)
    pic = Ts_X.cpu().data
    plt.imshow(pic[start+i+1][0], cmap='gray')
    plt.title('Test Image ' + str(i+1), fontsize = 15)
    plt.axis("off")

plt.figure(figsize = (23, 8))

for i in range(n_images):
    plt.subplot(1, n_images, i+1)
    pic = Ts_recon.cpu().data
    plt.imshow(pic[start+i+1][0], cmap='gray')
    plt.title('Reconstructed Image ' + str(i+1), fontsize = 15)
    plt.axis("off")


print("\n____________________________________________________\n")
