import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

class ModelTrainer:
    def __init__(self, data_dir="data/chest_xray", artifacts_dir="artifacts/", batch_size=32, epochs=5):
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Training execution engine bound to hardware: {self.device}")

    def initiate_model_training(self):
        try:
            train_path = os.path.join(self.data_dir, "train")
            test_path = os.path.join(self.data_dir, "test")
            
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Target directory structures missing at: {self.data_dir}")

            print("[INFO] Setting up image augmentations and pipeline transformations...")
            train_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            print(f"[INFO] Found {len(train_dataset)} training images mapped to classes: {train_dataset.classes}")
            print("[INFO] Initializing Modern MobileNetV3 Small Network...")
            
            # Using the exact up-to-date stable weights definition format
            weights = models.MobileNetV3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
            
            for param in model.parameters():
                param.requires_grad = False

            in_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 2)
            )
            model = model.to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

            print(f"[INFO] Launching deep model optimization for {self.epochs} epochs...")
            for epoch in range(self.epochs):
                model.train()
                running_loss = 0.0
                
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * images.size(0)

                epoch_loss = running_loss / len(train_dataset)
                print(f"Epoch {epoch+1}/{self.epochs} Completed -> Training Loss: {epoch_loss:.4f}")

            os.makedirs(self.artifacts_dir, exist_ok=True)
            model_save_path = os.path.join(self.artifacts_dir, "model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"[SUCCESS] Trained deep model state serialized to: '{model_save_path}'")
            return model_save_path

        except Exception as e:
            print(f"[ERROR] Exception hit during network training: {e}")
            raise e

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training()
