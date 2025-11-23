import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from sklearn.metrics import precision_score, recall_score, f1_score
from types import SimpleNamespace
import pandas as pd
from torchvision import models as torchvision_models
import timm

import argparse
import yaml
import logging
from datetime import datetime
import random
import time
from dataloader import prepare_data # 데이터 로딩 함수 임포트

try:
    from thop import profile
except ImportError:
    profile = None

from plot import plot_and_save_train_val_accuracy_graph, plot_and_save_val_accuracy_graph, plot_and_save_confusion_matrix, plot_and_save_f1_normal_graph, plot_and_save_loss_graph, plot_and_save_lr_graph, plot_and_save_compiled_graph

# =============================================================================
# 1. 로깅 및 모델 설정
# =============================================================================
def setup_logging(run_cfg, data_dir_name, baseline_model_name):
    """로그 파일을 log 폴더에 생성하고, 콘솔에도 함께 출력하도록 설정합니다."""
    show_log = getattr(run_cfg, 'show_log', True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not show_log:
        logging.disable(logging.CRITICAL)
        return '.', timestamp

    # 각 실행을 위한 고유한 디렉토리 생성 (baseline 모델 이름 포함)
    run_dir_name = f"baseline_{baseline_model_name}_{timestamp}"
    run_dir_path = os.path.join("log", data_dir_name, run_dir_name)
    os.makedirs(run_dir_path, exist_ok=True)
    
    log_filename = os.path.join(run_dir_path, f"log_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info(f"로그 파일이 '{log_filename}'에 저장됩니다.")
    return run_dir_path, timestamp

class Xie2019(nn.Module):
    def __init__(self, num_classes, dropout_rate = 0.6):
        super(Xie2019, self).__init__()
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            nn.Conv2d(3,64, 11, padding = 5, stride = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding = 1, stride = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 128, 3, padding = 1, stride = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Linear(128*8*8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_baseline_model(model_name, num_labels, pretrained):
    """지정된 이름의 torchvision 모델을 생성하고 마지막 레이어를 수정합니다."""
    logging.info(f"Baseline 모델 '{model_name}'을(를) 생성합니다 (사전 훈련 가중치: {'사용' if pretrained else '미사용'}).")
    
    if model_name == 'resnet18':
        model = torchvision_models.resnet18(weights=torchvision_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_labels)
    elif model_name == 'efficientnet_b0':
        model = torchvision_models.efficientnet_b0(weights=torchvision_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_labels)
    elif model_name == 'mobilenet_v4':
        # timm 라이브러리를 사용하여 MobileNetV4 모델을 생성합니다.
        # 'mobilenetv4_conv_small'은 가벼운 버전 중 하나입니다.
        # timm.create_model은 num_classes 인자를 통해 자동으로 마지막 분류 레이어를 교체해줍니다.
        model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, num_classes=num_labels)
    elif model_name == 'xie2019':
        # Xie2019 모델은 사전 훈련된 가중치를 지원하지 않습니다.
        model = Xie2019(num_classes=num_labels)
    elif model_name == 'vit':
        # timm 라이브러리를 사용하여 Vision Transformer 모델을 생성합니다.
        # 'vit_base_patch16_224'는 대표적인 ViT 모델입니다.
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_labels)
    else:
        raise ValueError(f"지원하지 않는 baseline 모델 이름입니다: {model_name}")
        
    return model

def log_model_parameters(model):
    """모델의 총 파라미터 수를 계산하고 로깅합니다."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("="*50)
    logging.info("모델 파라미터 수:")
    logging.info(f"  - 총 파라미터: {total_params:,} 개")
    logging.info(f"  - 학습 가능한 파라미터: {trainable_params:,} 개")
    logging.info("="*50)

# =============================================================================
# 2. 훈련 및 평가 함수
# =============================================================================
def evaluate(run_cfg, model, data_loader, device, criterion, loss_function_name, desc="Evaluating", class_names=None, log_class_metrics=False):
    """모델을 평가하고 정확도, 정밀도, 재현율, F1 점수를 로깅합니다."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    show_log = getattr(run_cfg, 'show_log', True)
    progress_bar = tqdm(data_loader, desc=desc, leave=False, disable=not show_log)
    with torch.no_grad():
        for images, labels, _ in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            
            if loss_function_name == 'bcewithlogitsloss':
                loss = criterion(outputs[:, 1].unsqueeze(1), labels.float().unsqueeze(1))
            else: # crossentropyloss
                loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if total == 0:
        logging.warning("평가 데이터가 없습니다. 평가를 건너뜁니다.")
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'loss': float('inf'), 'labels': [], 'preds': []}

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    
    if desc.startswith("[Valid]"):
        acc_label = "Val Acc"
        log_message = f'{desc} | Loss: {avg_loss:.4f} | {acc_label}: {accuracy:.2f}%'
    else:
        acc_label = "Test Acc"
        log_message = f'{desc} Loss: {avg_loss:.4f} | {acc_label}: {accuracy:.2f}%'
    logging.info(log_message)

    if log_class_metrics and class_names:
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        for i, class_name in enumerate(class_names):
            log_line = (f"[Metrics for '{class_name}'] | "
                        f"Precision: {precision_per_class[i]:.4f} | "
                        f"Recall: {recall_per_class[i]:.4f} | "
                        f"F1: {f1_per_class[i]:.4f}")
            logging.info(log_line)

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_per_class': f1_per_class if log_class_metrics and class_names else None,
        'labels': all_labels,
        'preds': all_preds
    }

def train(run_cfg, train_cfg, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names, pos_weight):
    """모델 훈련 및 검증을 수행하고 최고 성능 모델을 저장합니다."""
    logging.info("train 모드를 시작합니다.")
    model_path = os.path.join(run_dir_path, run_cfg.model_path)

    # --- 손실 함수 설정 ---
    loss_function_name = getattr(train_cfg, 'loss_function', 'CrossEntropyLoss').lower()
    if loss_function_name == 'bcewithlogitsloss':
        # BCEWithLogitsLoss는 [B, 1] 형태의 출력을 기대하므로 모델의 마지막 레이어 수정이 필요할 수 있습니다.
        # 이 코드에서는 num_labels=2를 가정하고, 출력을 [B, 2]에서 [B, 1]로 변환하여 사용합니다.
        
        # --- 모델 아키텍처에 따라 마지막 분류 레이어를 동적으로 찾기 ---
        last_layer = None
        if hasattr(model, 'fc'): # ResNet 계열
            last_layer = model.fc
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential) and isinstance(model.classifier[-1], nn.Linear): # EfficientNet 계열
            last_layer = model.classifier[-1]
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear): # timm으로 생성된 MobileNetV4 계열
            last_layer = model.classifier
        elif hasattr(model, 'head'): # timm으로 생성된 ViT 계열
            last_layer = model.head
        
        if last_layer is None:
            logging.warning("모델의 마지막 분류 레이어를 자동으로 찾을 수 없습니다. BCE 손실 함수 사용 시 num_labels 확인을 건너뜁니다.")
        elif last_layer.out_features != 2:
            logging.warning(f"BCE 손실 함수는 이진 분류(num_labels=2)에 최적화되어 있습니다. 현재 num_labels={last_layer.out_features}")
        
        weight_value = getattr(train_cfg, 'bce_pos_weight', None)
        if weight_value == 'auto':
            final_pos_weight = pos_weight.to(device) if pos_weight is not None else None
        else:
            final_pos_weight = torch.tensor(float(weight_value), dtype=torch.float).to(device) if weight_value is not None else None
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=final_pos_weight)
        logging.info(f"손실 함수: BCEWithLogitsLoss (pos_weight: {final_pos_weight.item() if final_pos_weight is not None else 'None'})")
    else: # 'crossentropyloss' 또는 기본값
        criterion = nn.CrossEntropyLoss()
        logging.info("손실 함수: CrossEntropyLoss")

    best_model_criterion = getattr(train_cfg, 'best_model_criterion', 'F1_average')
    best_metric = 0.0 if best_model_criterion != 'val_loss' else float('inf')

    for epoch in range(train_cfg.epochs):
        logging.info("-" * 50)
        model.train()
        if optimizer and hasattr(optimizer, 'train'):
            optimizer.train()

        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.epochs} [Training]", leave=False, disable=False)
        for images, labels, _ in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = model(images)

            if loss_function_name == 'bcewithlogitsloss':
                # BCEWithLogitsLoss는 [B, 1] 형태의 출력을 기대합니다.
                # outputs: [B, 2] -> [B, 1] (Defect 클래스에 대한 로짓만 사용)
                loss = criterion(outputs[:, 1].unsqueeze(1), labels.float().unsqueeze(1))
            else: # crossentropyloss
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            step_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{step_lr:.6f}")

        train_acc = 100 * correct / total
        logging.info(f'[Train] [{epoch+1}/{train_cfg.epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')
        
        eval_results = evaluate(run_cfg, model, valid_loader, device, criterion, loss_function_name, desc=f"[Valid] [{epoch+1}/{train_cfg.epochs}]", class_names=class_names, log_class_metrics=True)
        
        # 에포크 종료 시 Learning Rate 로깅
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        logging.info(f"[LR] [{epoch+1}/{train_cfg.epochs}] | Learning Rate: {current_lr:.6f}")

        current_metric = 0.0
        if best_model_criterion == 'val_loss':
            current_metric = eval_results['loss']
            is_best = current_metric < best_metric
        else: # F1 score variants
            if best_model_criterion == 'F1_Normal' and eval_results['f1_per_class'] is not None:
                current_metric = eval_results['f1_per_class'][0] # Assuming 'Normal' is the first class
            elif best_model_criterion == 'F1_Defect' and eval_results['f1_per_class'] is not None and len(eval_results['f1_per_class']) > 1:
                current_metric = eval_results['f1_per_class'][1] # Assuming 'Defect' is the second class
            else: # 'F1_average' or default
                current_metric = eval_results['f1_macro']
            is_best = current_metric > best_metric
        
        if is_best:
            best_metric = current_metric
            torch.save(model.state_dict(), model_path)
            criterion_name = best_model_criterion.replace('_', ' ')
            logging.info(f"[Best Model Saved] ({criterion_name}: {best_metric:.4f}) -> '{model_path}'")
        
        if scheduler:
            scheduler.step()

def inference(run_cfg, model_cfg, model, data_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=None):
    """저장된 모델로 추론 및 성능 평가를 수행합니다."""
    logging.info(f"{mode_name} 모드를 시작합니다.")
    model_path = os.path.join(run_dir_path, run_cfg.model_path)
    if not os.path.exists(model_path):
        logging.error(f"모델 파일('{model_path}')을 찾을 수 없습니다. 'train' 모드를 먼저 실행했는지 확인하세요.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logging.info(f"'{model_path}' 가중치 로드 완료.")
    except Exception as e:
        logging.error(f"모델 가중치 로딩 중 오류 발생: {e}")
        return

    model.eval()

    # --- 성능 지표 측정 ---
    gflops_per_sample = 0.0
    try:
        sample_image, _, _ = data_loader.dataset.dataset[0] if isinstance(data_loader.dataset, Subset) else data_loader.dataset[0]
        dummy_input = sample_image.unsqueeze(0).to(device)

        if profile:
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
            gmacs = macs / 1e9
            gflops_per_sample = (macs * 2) / 1e9
            logging.info(f"연산량 (MACs): {gmacs:.2f} GMACs per sample")
            logging.info(f"연산량 (FLOPs): {gflops_per_sample:.2f} GFLOPs per sample")
        else:
            logging.info("연산량 (FLOPs): N/A (thop 라이브러리 미설치)")
    except Exception as e:
        logging.error(f"FLOPS 측정 중 오류 발생: {e}")

    # --- 샘플 당 Forward Pass 시간 및 메모리 사용량 측정 ---
    avg_inference_time_per_sample = 0.0
    logging.info("GPU 캐시를 비우고, 샘플 당 Forward Pass 시간 및 최대 GPU 메모리 사용량 측정을 시작합니다...")
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # 시간 측정을 위한 예열(warm-up)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # 실제 시간 측정
        num_iterations = 100
        total_time = 0.0
        with torch.no_grad():
            for _ in range(num_iterations):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                total_time += start_event.elapsed_time(end_event) # ms
        
        avg_inference_time_per_sample = total_time / num_iterations

        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        logging.info(f"샘플 당 평균 Forward Pass 시간: {avg_inference_time_per_sample:.2f}ms ({num_iterations}회 반복)")
        logging.info(f"샘플 당 Forward Pass 시 최대 GPU 메모리 사용량: {peak_memory_mb:.2f} MB")
    else:
        logging.info("CUDA를 사용할 수 없어 CPU 추론 시간을 측정합니다.")

        # CPU 시간 측정을 위한 예열(warm-up)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # 실제 시간 측정
        num_iterations = 100
        total_time = 0.0
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                total_time += (end_time - start_time) * 1000 # ms

        avg_inference_time_per_sample = total_time / num_iterations
        logging.info(f"샘플 당 평균 Forward Pass 시간 (CPU): {avg_inference_time_per_sample:.2f}ms ({num_iterations}회 반복)")
    # --- 평가 또는 순수 추론 ---
    logging.info("테스트 데이터셋에 대한 추론을 시작합니다...")
    only_inference_mode = getattr(run_cfg, 'only_inference', False)

    if only_inference_mode:
        all_filenames, all_predictions, all_confidences = [], [], []
        progress_bar = tqdm(data_loader, desc=f"[{mode_name}]", leave=False, disable=not getattr(run_cfg, 'show_log', True))
        with torch.no_grad():
            for images, _, filenames in progress_bar:
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted_indices = torch.max(probabilities, 1)
                all_filenames.extend(filenames)
                all_predictions.extend([class_names[p] for p in predicted_indices.cpu().numpy()])
                all_confidences.extend(confidences.cpu().numpy())
        
        results_df = pd.DataFrame({'filename': all_filenames, 'prediction': all_predictions, 'confidence': all_confidences})
        results_df['confidence'] = results_df['confidence'].map('{:.4f}'.format)
        result_csv_path = os.path.join(run_dir_path, f'inference_results_{timestamp}.csv')
        results_df.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"추론 결과가 '{result_csv_path}'에 저장되었습니다.")
        final_acc = None
    else:
        # 추론 시에는 간단한 손실 함수를 임시로 생성하여 전달합니다.
        loss_function_name = getattr(SimpleNamespace(**yaml.safe_load(open('config.yaml', 'r', encoding='utf-8'))['training_baseline']), 'loss_function', 'CrossEntropyLoss').lower()
        if loss_function_name == 'bcewithlogitsloss':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        eval_results = evaluate(run_cfg, model, data_loader, device, criterion, loss_function_name, desc=f"[{mode_name}]", class_names=class_names, log_class_metrics=True)
        final_acc = eval_results['accuracy']

        if eval_results['labels'] and eval_results['preds']:
            plot_and_save_confusion_matrix(eval_results['labels'], eval_results['preds'], class_names, run_dir_path, timestamp)
    
    return final_acc

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="YAML 설정을 이용한 Baseline 모델 분류기")
    parser.add_argument('--config', type=str, default='config.yaml', help="설정 파일 경로. 기본값: 'config.yaml'")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    run_cfg = SimpleNamespace(**config['run'])
    train_cfg = SimpleNamespace(**config['training_baseline'])
    model_cfg = SimpleNamespace(**config['model'])
    baseline_cfg = SimpleNamespace(**config.get('baseline', {})) # baseline 섹션 로드
    run_cfg.dataset = SimpleNamespace(**run_cfg.dataset)
    
    # --- 전역 시드 고정 ---
    global_seed = getattr(run_cfg, 'global_seed', None)
    if global_seed is not None:
        random.seed(global_seed)
        os.environ['PYTHONHASHSEED'] = str(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        torch.cuda.manual_seed(global_seed)
        logging.info(f"전역 랜덤 시드를 {global_seed}로 고정합니다.")

    # Baseline 모델 이름 확인
    baseline_model_name = getattr(baseline_cfg, 'model_name', 'resnet18')

    # --- 로깅 및 디렉토리 설정 ---
    data_dir_name = run_cfg.dataset.name
    if run_cfg.mode == 'train':
        run_dir_path, timestamp = setup_logging(run_cfg, data_dir_name, baseline_model_name)
    elif run_cfg.mode == 'inference':
        run_dir_path = getattr(run_cfg, 'only_inference_dir', None)
        if getattr(run_cfg, 'show_log', True) and (not run_dir_path or not os.path.isdir(run_dir_path)):
            logging.error("추론 모드에서는 'config.yaml'에 'only_inference_dir'를 올바르게 설정해야 합니다.")
            exit()
        _, timestamp = setup_logging(run_cfg, data_dir_name, baseline_model_name)
    
    config_str = yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logging.info("="*50)
    logging.info("config.yaml:")
    logging.info("\n" + config_str)
    logging.info("="*50)
    
    use_cuda_if_available = getattr(run_cfg, 'cuda', True)
    device = torch.device("cuda" if use_cuda_if_available and torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        logging.info(f"CUDA 사용 가능. GPU 사용을 시작합니다. (Device: {torch.cuda.get_device_name(0)})")
    else:
        if use_cuda_if_available:
            logging.warning("config.yaml에서 CUDA 사용이 활성화되었지만, 사용 가능한 CUDA 장치를 찾을 수 없습니다. CPU를 사용합니다.")
        logging.info("CPU 사용을 시작합니다.")

    # --- 데이터 준비 ---
    train_loader, valid_loader, test_loader, num_labels, class_names, pos_weight = prepare_data(run_cfg, train_cfg, model_cfg)

    # --- Baseline 모델 생성 ---
    model = create_baseline_model(baseline_model_name, num_labels, pretrained=train_cfg.pre_trained).to(device)
    log_model_parameters(model)
    
    # --- 옵티마이저 및 스케줄러 설정 ---
    optimizer, scheduler = None, None
    if run_cfg.mode == 'train':
        if getattr(train_cfg, 'optimizer', 'adamw').lower() == 'sgd':
            logging.info(f"옵티마이저: SGD (lr={train_cfg.lr}, momentum={train_cfg.momentum}, weight_decay={train_cfg.weight_decay})")
            optimizer = optim.SGD(model.parameters(), lr=train_cfg.lr, momentum=train_cfg.momentum, weight_decay=train_cfg.weight_decay)
        else:
            logging.info(f"옵티마이저: AdamW (lr={train_cfg.lr})")
            optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr)

        # scheduler_params가 없으면 빈 객체로 초기화
        scheduler_params = getattr(train_cfg, 'scheduler_params', SimpleNamespace())

        if getattr(train_cfg, 'scheduler', 'none').lower() == 'multisteplr':
            milestones = getattr(train_cfg, 'milestones', [30, 60, 80])
            gamma = getattr(train_cfg, 'gamma', 0.1)
            logging.info(f"스케줄러: MultiStepLR (milestones={milestones}, gamma={gamma})")
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif getattr(train_cfg, 'scheduler', 'none').lower() == 'cosineannealinglr':
            T_max = getattr(scheduler_params, 'T_max', train_cfg.epochs)
            eta_min = getattr(scheduler_params, 'eta_min', 0.0)
            logging.info(f"스케줄러: CosineAnnealingLR (T_max={T_max}, eta_min={eta_min})")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        else:
            logging.info("스케줄러를 사용하지 않습니다.")
        logging.info("="*50)

    # --- 모드에 따라 실행 ---
    if run_cfg.mode == 'train':
        train(run_cfg, train_cfg, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names, pos_weight)
        logging.info("="*50)
        logging.info("훈련 완료. 최종 모델 성능을 테스트 세트로 평가합니다.")
        final_acc = inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Test", class_names=class_names)

        log_filename = f"log_{timestamp}.log"
        log_file_path = os.path.join(run_dir_path, log_filename)
        if final_acc is not None:
            plot_and_save_val_accuracy_graph(log_file_path, run_dir_path, final_acc, timestamp)
            plot_and_save_train_val_accuracy_graph(log_file_path, run_dir_path, final_acc, timestamp)
            plot_and_save_f1_normal_graph(log_file_path, run_dir_path, timestamp, class_names)
            plot_and_save_loss_graph(log_file_path, run_dir_path, timestamp)
            plot_and_save_lr_graph(log_file_path, run_dir_path, timestamp)
            plot_and_save_compiled_graph(run_dir_path, timestamp)

    elif run_cfg.mode == 'inference':
        inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=class_names)

if __name__ == '__main__':
    main()