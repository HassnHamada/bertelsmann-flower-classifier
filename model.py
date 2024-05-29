import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image


class Model:
    def __init__(self,  gpu: bool, /, *, save_dir: str = None, epochs: int = None):
        self.train_test = False
        self.save_dir = save_dir
        self.epochs = epochs
        self.gpu = gpu
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def new(self, arch: str, learning_rate: float, hidden_units: int, out_features: int):
        self.arch = arch
        self.hidden_units = hidden_units
        self.out_features = out_features
        self.valid_loss_min = np.Inf
        self.model = models.get_model(self.arch, weights=models.get_model_weights(self.arch))

        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier.extend(nn.Sequential(nn.ReLU(inplace=True),
                                                   nn.Dropout(p=0.5, inplace=False),
                                                   nn.Linear(in_features=self.model.classifier[-1].out_features,
                                                             out_features=self.hidden_units, bias=True),
                                                   nn.ReLU(inplace=True),
                                                   nn.Dropout(p=0.5, inplace=False),
                                                   nn.Linear(in_features=self.hidden_units,
                                                             out_features=self.out_features, bias=True),
                                                   nn.LogSoftmax(dim=1)))

        self.criterion = nn.NLLLoss()

        self.optimizer = optim.SGD(self.model.classifier.parameters(),
                                   lr=learning_rate)

    def load_data(self, data_dir: str):
        self.train_test = True
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        batch_size = 64

        train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406),
                                                                    (0.229, 0.224, 0.225))])
        valid_test_transforms = transforms.Compose([transforms.Resize(224),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                                         (0.229, 0.224, 0.225))])

        train_datasets = datasets.ImageFolder(train_dir,
                                              transform=train_transforms)
        valid_datasets = datasets.ImageFolder(valid_dir,
                                              transform=valid_test_transforms)
        test_datasets = datasets.ImageFolder(test_dir,
                                             transform=valid_test_transforms)

        self.train_dataloaders = DataLoader(train_datasets,
                                            batch_size=batch_size,
                                            shuffle=True)
        self.valid_dataloaders = DataLoader(valid_datasets,
                                            batch_size=batch_size,
                                            shuffle=True)
        self.test_dataloaders = DataLoader(test_datasets,
                                           batch_size=batch_size,
                                           shuffle=True)
        self.idx_to_class = dict((v, k) for k, v in train_datasets.class_to_idx.items())
        assert self.out_features == len(train_datasets.classes)

    def train(self):
        assert self.train_test, "No training data found"
        assert self.epochs is not None, "No epochs specified"
        print(f"Training for {self.epochs} epochs...")
        if self.gpu:
            self.model.cuda()
        else:
            self.model.cpu()

        for epoch in range(self.epochs):
            train_loss = 0.0
            valid_loss = 0.0

            self.model.train()
            for data, target in self.train_dataloaders:
                if self.gpu:
                    data, target = data.cuda(), target.cuda()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                print(".", end='')
            print()

            self.model.eval()
            for data, target in self.valid_dataloaders:
                if self.gpu:
                    data, target = data.cuda(), target.cuda()
                output = self.model(data)
                loss = self.criterion(output, target)
                valid_loss += loss.item()
                print(".", end='')
            print()

            train_loss = train_loss/len(self.train_dataloaders)
            valid_loss = valid_loss/len(self.valid_dataloaders)

            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')
            if self.save_dir is not None:
                if valid_loss < self.valid_loss_min:
                    self.valid_loss_min = valid_loss
                    self.save(f'{self.save_dir}/best.pt')
                print(f'Saving checkpoint {self.save_dir}/checkpoint_{epoch:03d}_{valid_loss:.6f}.pt ...')
                self.save(f'{self.save_dir}/checkpoint_{epoch:03d}_{valid_loss:.6f}.pt')

    def test(self):
        assert self.train_test, "No testing data found"
        print("Testing...")
        if self.gpu:
            self.model.cuda()
        else:
            self.model.cpu()
        total = correct = 0
        self.model.eval()
        for data, target in self.test_dataloaders:
            if self.gpu:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            _, pred = torch.max(output, 1)
            total += pred.shape[0]
            correct += (pred == target).sum()
            print(".", end='')
        print()
        accuracy = correct / total
        print(f"Accuracy: {accuracy.item()}")

    def save(self, path):
        checkpoint = {
            # Needed for recreating the model
            'arch': self.arch,
            'hidden_units': self.hidden_units,
            'out_features': self.out_features,
            'state_dict': self.model.state_dict(),
            # Needed for training
            'optimizer': self.optimizer,
            'criterion': self.criterion,
            'valid_loss_min': self.valid_loss_min,
            # Needed for prediction
            'idx_to_class': self.idx_to_class,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.arch = checkpoint['arch']
        self.hidden_units = checkpoint['hidden_units']
        self.out_features = checkpoint['out_features']
        self.model = models.get_model(self.arch,
                                      weights=models.get_model_weights(self.arch))
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier.extend(nn.Sequential(nn.ReLU(inplace=True),
                                                   nn.Dropout(p=0.5, inplace=False),
                                                   nn.Linear(in_features=self.model.classifier[-1].out_features,
                                                             out_features=self.hidden_units, bias=True),
                                                   nn.ReLU(inplace=True),
                                                   nn.Dropout(p=0.5, inplace=False),
                                                   nn.Linear(in_features=self.hidden_units,
                                                             out_features=self.out_features, bias=True),
                                                   nn.LogSoftmax(dim=1)))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer = checkpoint['optimizer']
        self.criterion = checkpoint['criterion']
        self.valid_loss_min = checkpoint['valid_loss_min']
        self.idx_to_class = checkpoint['idx_to_class']

    def predict(self, image_path, topk=1):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        def process_image(image: Image.Image) -> np.ndarray:
            ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
                returns an Numpy array
            '''

            # TODO: Process a PIL image for use in a PyTorch model
            w, h = image.size
            if w < h:
                image.thumbnail((256, int(256*h/w)), Image.Resampling.LANCZOS)
            else:
                image.thumbnail((int(256*w/h), 256), Image.Resampling.LANCZOS)

            left = (image.width - 224)/2
            top = (image.height - 224)/2
            right = left + 224
            bottom = top + 224

            image = image.crop((left, top, right, bottom))

            np_image = np.array(image) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            np_image = (np_image - mean) / std
            np_image = np_image.transpose((2, 0, 1)).reshape((1, 3, 224, 224))
            assert np_image.shape == (1, 3, 224, 224)
            return np_image

        # Load the image
        with Image.open(image_path) as image:
            image = process_image(image)
            image = torch.from_numpy(image).type(torch.FloatTensor)

        if self.gpu:
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.eval()

        # Get the predictions
        with torch.no_grad():
            if self.gpu:
                image = image.cuda()
            output = self.model(image)
            ps = torch.exp(output)
            probs, indices = ps.topk(topk)
            if self.gpu:
                probs = probs.cpu().numpy()[0]
                indices = indices.cpu().numpy()[0]
            else:
                probs = probs.numpy()[0]
                indices = indices.numpy()[0]

        classes = [self.idx_to_class[idx] for idx in indices]

        return probs.tolist(), classes
