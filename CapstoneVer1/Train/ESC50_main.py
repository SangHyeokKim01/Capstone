from ESC50Dataset import ESC50Dataset
import models
import torch
from trainer import Trainer
from torch.utils.data import DataLoader


def main():
    EPOCHS= 1000
    BATCH = 1000
    LR = 2e-3
    segment_type = 'long'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    csv_path = "./ESC-50/meta/esc50.csv"


    train_dataset = ESC50Dataset(
        csv_file=csv_path,
        npy_dir='./preprocessed_mel', #오디오 wav 파일을 프리프로세싱 해놓은 데이터 (preprocess_esc.py에 의해 실행됨)
        fold=[1, 3, 4, 5],
        segment_type=segment_type  
    )

    val_dataset = ESC50Dataset(
        csv_file=csv_path,
        npy_dir='./preprocessed_mel', #오디오 wav 파일을 프리프로세싱 해놓은 데이터
        fold=[2],
        segment_type=segment_type
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=8, pin_memory=True)


    
    sample_input, _ = next(iter(train_loader))  # (B, C, H, W)
    input_shape = sample_input.shape[1:]        # (C, H, W)
    print(input_shape)
    num_classes = len(set(train_dataset.meta['target']))
    

    mymodel = models.ESC50_CNN_Ver1(
                    num_classes=num_classes,
                    input_shape=input_shape,
                    conv1_channels=80,
                    conv2_channels=80,
                    fc1_size=5000,
                    fc2_size=5000,
                    dropout=0.5
                    )
    
    trainer = Trainer(num_epochs=EPOCHS, 
                      model=mymodel, 
                      train_loader=train_loader, 
                      val_loader=val_loader, 
                      device=device,
                      lr = LR)


    

    trainer.training() # 학습 
    
    # 저장
    trainer.save_model(
        filename=f"ESC50_CNN_{segment_type}_{EPOCHS}_{BATCH}.pt",
        save_dir="trained_model"
    )


if __name__ == "__main__":
    main()
