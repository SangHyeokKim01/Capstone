import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

class Trainer:
    """
      - SGD + Nesterov momentum = 0.9
      - Weight decay = 0.001
      - (Short variant) lr=0.002, epochs=300
      - (Long variant) lr=0.01, epochs=150
    """
    def __init__(self, num_epochs, model, train_loader, val_loader=None,
                 device='cuda', lr=0.002, weight_decay=1e-4, momentum=0.9):
        self.Num_epochs = num_epochs
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss() #사용한 LF 이건지 모르겠음

        # 논문에서 언급된 SGD + Nesterov momentum = 0.9, weight_decay=0.001
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay
        )

        # self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def training(self):
          # 모델을 학습 모드로
        global_starttime = time.time()
        max_val_acc = 0
        max_val_acc_idx = 0
        for epoch in range(self.Num_epochs):
            self.model.train()
            start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                # 정확도 계산
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            # 검증 (val_loader 있으면)
            val_acc = self.evaluate() if self.val_loader else 0.0

            end_time = time.time()
            epoch_time = end_time - start_time
            accumulated_time = end_time - global_starttime

            print(f"[Epoch {epoch+1}/{self.Num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.2f}s | total {accumulated_time:.2f}s")
            
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_acc_idx = epoch + 1
        print(f"max_val_acc_epoch: {max_val_acc_idx} | max_val_acc: {max_val_acc}")



    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        self.model.train()  # 다시 학습 모드로 전환
        return correct / total
    
    
    def save_model(self, filename="trained_model.pt", save_dir="trained_model"):
        # 디렉토리 없으면 자동 생성
        os.makedirs(save_dir, exist_ok=True)

        full_path = os.path.join(save_dir, filename)
        torch.save(self.model, full_path)
        print(f"[INFO] Model saved to: {full_path}")
