# 파일명: train_efficientnet_b0.py

import os
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
from PIL import Image

# =============================================================================
# 1. 설정 (이곳에서 모든 변수를 수정하세요)
# =============================================================================
# --- 데이터 경로 설정 ---
TRAIN_IMG_DIR = 'D:/Sewer-ML/Train'
VALID_IMG_DIR = 'D:/Sewer-ML/Valid'
TEST_IMG_DIR = 'D:/Sewer-ML/Valid'  # 테스트셋으로 검증셋 사용
TRAIN_CSV = 'D:/Sewer-ML/SewerML_Train.csv'
VALID_CSV = 'D:/Sewer-ML/SewerML_Val.csv'
TEST_CSV = 'D:/Sewer-ML/SewerML_Val.csv'    # 테스트셋으로 검증셋 사용

# --- 모델 및 이미지 설정 ---
MODEL_NAME = 'efficientnet_b0'
IMG_SIZE = 224  # EfficientNet-B0의 기본 이미지 크기
IN_CHANNELS = 3
NUM_CLASSES = 2 # 'Normal', 'Defect' 이진 분류

# --- 훈련 하이퍼파라미터 ---
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# --- 실행 환경 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 로그 및 모델 저장 경로는 train_model 함수 내에서 동적으로 생성됩니다.
LOG_BASE_DIR = 'log/Sewer-ML'
MODEL_FILENAME = 'best_model.pth'

# --- 최고 모델 저장 기준 설정 ---
# 옵션: 'F1_Macro', 'F1_Normal', 'F1_Defect'
# F1_Normal: 'Normal' 클래스의 F1 점수가 가장 높을 때 모델을 저장합니다.
BEST_MODEL_CRITERION = 'F1_Normal'

# =============================================================================
# 2. 커스텀 데이터셋 정의
# =============================================================================
class CustomImageDataset(Dataset):
    """CSV 파일과 이미지 폴더 경로를 받아 데이터를 로드하는 커스텀 데이터셋입니다."""
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"경고: 파일을 찾을 수 없습니다 - {img_path}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color = 'black')

        if self.transform:
            image = self.transform(image)

        label = int(self.img_labels.loc[idx, 'Defect'])
        return image, label, img_name

# =============================================================================
# 3. 훈련 및 평가 함수
# =============================================================================
def setup_logging(log_base_dir):
    """로그 파일을 log 폴더에 생성하고, 콘솔에도 함께 출력하도록 설정합니다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"run_{timestamp}"
    run_dir_path = os.path.join(log_base_dir, run_dir_name)
    os.makedirs(run_dir_path, exist_ok=True)
    
    log_filename = os.path.join(run_dir_path, f"log_{timestamp}.log")
    
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"로그 파일이 '{log_filename}'에 저장됩니다.")
    return run_dir_path

def evaluate(model, data_loader, device, desc="Evaluating", class_names=None, log_class_metrics=False):
    """모델을 평가하고 다양한 성능 지표를 반환 및 로깅합니다."""
    model.eval()
    all_preds = []
    all_labels = []
    total_forward_time = 0.0
    
    progress_bar = tqdm(data_loader, desc=desc, leave=False)
    with torch.no_grad():
        for images, labels, _ in progress_bar:
            images, labels = images.to(device), labels.to(device)

            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                outputs = model(images)
                end_event.record()
                torch.cuda.synchronize()
                total_forward_time += start_event.elapsed_time(end_event) / 1000.0
            else:
                start_time = time.time()
                outputs = model(images)
                end_time = time.time()
                total_forward_time += (end_time - start_time)

            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels:
        return {'f1_macro': 0.0, 'f1_per_class': [0.0, 0.0], 'forward_time': 0.0}

    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    log_message = f'{desc} | F1: {f1_macro:.4f} | Precision: {precision_macro:.4f} | Recall: {recall_macro:.4f}'
    logging.info(log_message)

    f1_per_class = None
    if log_class_metrics and class_names:
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        logging.info("-" * 30)
        for i, class_name in enumerate(class_names):
            logging.info(f"  - F1 Score for '{class_name}': {f1_per_class[i]:.4f}")
        logging.info("-" * 30)

    return {
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'forward_time': total_forward_time
    }

def train_model():
    """전체 훈련 및 평가 파이프라인을 실행합니다."""
    run_dir_path = setup_logging(LOG_BASE_DIR)
    model_save_path = os.path.join(run_dir_path, MODEL_FILENAME)

    logging.info("="*50)
    logging.info(f"모델 훈련 시작: {MODEL_NAME}")
    logging.info(f"사용 장치: {DEVICE}")
    logging.info(f"로그 및 모델 저장 경로: {run_dir_path}")
    logging.info("="*50)

    # --- 데이터 전처리 및 로더 준비 ---
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        normalize
    ])
    valid_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    class_names = ['Normal', 'Defect']
    train_dataset = CustomImageDataset(csv_file=TRAIN_CSV, img_dir=TRAIN_IMG_DIR, transform=train_transform)
    valid_dataset = CustomImageDataset(csv_file=VALID_CSV, img_dir=VALID_IMG_DIR, transform=valid_test_transform)
    test_dataset = CustomImageDataset(csv_file=TEST_CSV, img_dir=TEST_IMG_DIR, transform=valid_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    logging.info(f"데이터 로드 완료: Train {len(train_dataset)}개, Valid {len(valid_dataset)}개, Test {len(test_dataset)}개")

    # --- 모델, 손실 함수, 옵티마이저 정의 ---
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES) # 마지막 레이어를 교체
    model.to(DEVICE)

    # --- 학습 가능한 파라미터 수 계산 및 로깅 ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("="*50)
    logging.info(f"학습 가능한 총 파라미터 수: {total_params:,} 개")
    logging.info("="*50)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- 훈련 루프 ---
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]", leave=False)
        for images, labels, _ in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        
        eval_results = evaluate(model, valid_loader, DEVICE, desc=f"Epoch {epoch+1}/{EPOCHS} [Validating]", class_names=class_names, log_class_metrics=True)
        logging.info(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_loss:.4f}")
        
        # --- 최고 성능 모델 저장 기준 선택 ---
        current_f1 = 0.0
        criterion_name = "F1 Macro"
        if BEST_MODEL_CRITERION == 'F1_Normal' and eval_results['f1_per_class'] is not None:
            current_f1 = eval_results['f1_per_class'][0] # 'Normal' 클래스는 인덱스 0
            criterion_name = "F1 Normal"
        elif BEST_MODEL_CRITERION == 'F1_Defect' and eval_results['f1_per_class'] is not None:
            current_f1 = eval_results['f1_per_class'][1] # 'Defect' 클래스는 인덱스 1
            criterion_name = "F1 Defect"
        else: # 'F1_Macro' 또는 그 외
            current_f1 = eval_results['f1_macro']
        
        # 최고 성능 모델 저장
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"  -> 최고 성능 모델 저장 완료 ({criterion_name}: {best_f1:.4f}) -> '{model_save_path}'")
        
        scheduler.step()

    # --- 최종 테스트 ---
    logging.info("\n" + "="*50)
    logging.info("훈련 완료. 최종 모델로 추론(Inference)을 시작합니다...")
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    
    # GPU 메모리 사용량 측정
    dummy_input = torch.randn(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE).to(DEVICE)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        with torch.no_grad():
            _ = model(dummy_input)
        peak_memory_bytes = torch.cuda.max_memory_allocated(DEVICE)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        logging.info(f"샘플 당 Forward Pass 시 최대 GPU 메모리 사용량: {peak_memory_mb:.2f} MB")
    
    # 최종 성능 평가
    test_results = evaluate(model, test_loader, DEVICE, desc="[Final Test]", class_names=class_names, log_class_metrics=True)
    
    num_test_samples = len(test_loader.dataset)
    total_time = test_results['forward_time']
    avg_time_ms = (total_time / num_test_samples) * 1000 if num_test_samples > 0 else 0
    logging.info(f"총 forward pass 시간: {total_time:.2f}s (테스트 샘플 {num_test_samples}개)")
    logging.info(f"샘플 당 평균 forward pass 시간: {avg_time_ms:.2f}ms")
    logging.info("="*50)

# =============================================================================
# 4. 메인 실행 블록
# =============================================================================
if __name__ == '__main__':
    train_model()
