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

print("\n...............................................")

print("\nLoading Dataset into DataLoader...")

# Get All Images
Cat_Imgs = os.path.join(data_path, 'Cat/')
Cat_Imgs = glob.glob(Cat_Imgs + '*')

Dog_Imgs = os.path.join(data_path, 'Dog/')
Dog_Imgs = glob.glob(Dog_Imgs + '*')

all_imgs = Cat_Imgs + Dog_Imgs
shuffle(all_imgs)

# Train - Validation - Test Split
train_imgs = all_imgs[:1600]
val_imgs = all_imgs[1600:1900]
test_imgs = all_imgs[1900:] 

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
Val_DL = torch.utils.data.DataLoader(imagePrep(val_imgs, dataset_transform), batch_size= batch_size)
Ts_DL = torch.utils.data.DataLoader(imagePrep(test_imgs, dataset_transform), batch_size= batch_size)

print("\nDataLoader Set!")
print("\n...............................................\n")

print("\nBuilding Convolutional Neural Network Model...")

# Define Convolutional Neural Network
class ConvNNet(torch.nn.Module):
    def __init__(self):
        super(ConvNNet, self).__init__()
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels= 1, out_channels= 64, kernel_size= (5,5)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size= 2),

            torch.nn.Conv2d(in_channels= 64, out_channels= 256 , kernel_size= (3,3)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size= 2),
        
            torch.nn.Conv2d(in_channels= 256, out_channels= 1024, kernel_size= (3,3)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size= 2)
        )

        in_feat = 1024 * 14 * 14

        self.fconn_net = torch.nn.Sequential(
            torch.nn.Linear(in_features= in_feat, out_features= 512),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(in_features= 512, out_features= 64),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(in_features= 64, out_features= 16),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(in_features= 16, out_features= 2),
            torch.nn.Sigmoid()
        )


    def forward(self, X):
        feat_extr = self.conv_net(X)
        flat = feat_extr.view(feat_extr.shape[0], -1)
        preds = self.fconn_net(flat)
        return preds

print("\nConvolutional Neural Network Model Set!")
print("\n...............................................\n")

print("\n____________________________________________________\n")

# defining the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining the model
cnn_model = ConvNNet().to(device)

# defining the optimizer
optimizer = torch.optim.Adam(cnn_model.parameters(), lr= learning_rate, weight_decay= weight_decay)

# defining the loss function
loss_function = torch.nn.CrossEntropyLoss().to(device)

print(cnn_model)
print("____________________________________________________\n")

print("\nTraining the CNN Model on Train and Validation Data...")

# Training and Validation of Model
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

for epoch in range(epochs):        
    epoch_loss = 0
    epoch_acc = 0

    for X, y in Tr_DL:
        img = X.to(device)
        y = y.to(device)

        img = torch.autograd.Variable(img)
    
        predictns = cnn_model(img)

        loss = loss_function(predictns, y)

        # Backward Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accu = ((predictns.argmax(dim=1) == y).float().mean())
        epoch_acc += accu
        epoch_loss += loss
        print('-', end= "", flush= True)

    epoch_acc = epoch_acc/len(Tr_DL)
    train_accuracy.append(epoch_acc)

    epoch_loss = epoch_loss/len(Tr_DL)
    train_loss.append(epoch_loss)

    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_acc = 0

        for val_X, val_y in Val_DL:
            val_X = val_X.to(device)
            val_y = val_y.to(device)

            val_preds = cnn_model(val_X)
            val_los = loss_function(val_preds, val_y)

            val_accu = ((val_preds.argmax(dim=1) == val_y).float().mean())
            val_epoch_acc += val_accu
            val_epoch_loss += val_los

        val_epoch_acc = val_epoch_acc/len(Val_DL)
        val_accuracy.append(val_epoch_acc)

        val_epoch_loss = val_epoch_loss/len(Val_DL)
        val_loss.append(val_epoch_loss)

        print("\nEpoch: {} | Train loss: {:.4f}  Train accuracy: {:.4f}  Validation loss: {:.4f}  Validation accuracy: {:.4f}".format(epoch+1, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc))

print("\n____________________________________________________\n")

# plotting the training and validation loss
fig = plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend()
plt.show()

print("\n____________________________________________________\n")

# plotting the training and validation Accuracy
fig = plt.figure(figsize=(10, 6))
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracies')
plt.legend()
plt.show()

print("\n____________________________________________________\n")

# Testing On Test Data
with torch.no_grad():
    Ts_acc = 0

    for Ts_X, Ts_y in Ts_DL:
        Ts_X = Ts_X.to(device)
        Ts_y = Ts_y.to(device)

        Ts_preds = cnn_model(Ts_X)
        Ts_accu = ((Ts_preds.argmax(dim=1) == Ts_y).float().mean())
        Ts_acc += Ts_accu

    Ts_acc = Ts_acc/len(Ts_DL)
    print("\n____________________________________________________")
    print("\nAccuracy on Test Data: {:.4f}\n".format(Ts_acc))
    print("\n____________________________________________________")
