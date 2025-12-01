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

try:
    import onnxruntime
    from onnx_utils import evaluate_onnx, measure_onnx_performance, measure_model_flops
except ImportError:
    onnxruntime = None

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
    elif loss_function_name == 'crossentropyloss':
        label_smoothing = getattr(train_cfg, 'label_smoothing', 0.0)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        logging.info(f"손실 함수: CrossEntropyLoss (label_smoothing: {label_smoothing})")
    else: # 'crossentropyloss' 또는 기본값
        raise ValueError(f"baseline.py에서 지원하지 않는 손실 함수입니다: {loss_function_name}")

    best_model_criterion = getattr(train_cfg, 'best_model_criterion', 'F1_average')
    best_metric = 0.0 if best_model_criterion != 'val_loss' else float('inf')

    # --- Warmup 설정 ---
    warmup_cfg = getattr(train_cfg, 'warmup', None)
    use_warmup = warmup_cfg and getattr(warmup_cfg, 'enabled', False)
    if use_warmup:
        warmup_epochs = getattr(warmup_cfg, 'epochs', 0)
        warmup_start_lr = getattr(warmup_cfg, 'start_lr', 0.0)
        warmup_end_lr = train_cfg.lr # Warmup 종료 LR은 메인 LR로 설정
        logging.info(f"Warmup 활성화: {warmup_epochs} 에포크 동안 LR을 {warmup_start_lr}에서 {warmup_end_lr}로 선형 증가시킵니다.")

        # Warmup 기간 동안에는 스케줄러를 비활성화합니다.
        original_scheduler_step = scheduler.step if scheduler else lambda: None
        if scheduler:
            scheduler.step = lambda: None

    for epoch in range(train_cfg.epochs):
        logging.info("-" * 50)
        model.train()
        if optimizer and hasattr(optimizer, 'train'):
            optimizer.train()

        running_loss = 0.0
        correct = 0
        total = 0
        
        # --- Warmup LR 조정 ---
        if use_warmup and epoch < warmup_epochs:
            if warmup_epochs > 1:
                lr_step = (warmup_end_lr - warmup_start_lr) / (warmup_epochs - 1)
                current_lr = warmup_start_lr + epoch * lr_step
            else: # warmup_epochs가 1인 경우
                current_lr = warmup_end_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # 에포크 시작 시 Learning Rate 로깅
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"[LR]    [{epoch+1}/{train_cfg.epochs}] | Learning Rate: {current_lr:.6f}")

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.epochs} [Training]", leave=False, disable=not getattr(run_cfg, 'show_log', True))
        for images, labels, _ in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = model(images)
            if loss_function_name == 'bcewithlogitsloss':
                loss = criterion(outputs[:, 1].unsqueeze(1), labels.float().unsqueeze(1))
            else: # crossentropyloss
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            running_loss += loss_val

            _, predicted = torch.max(outputs.data, 1) # outputs는 로짓이므로 그대로 사용
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            step_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{step_lr:.6f}")

        train_acc = 100 * correct / total
        logging.info(f'[Train] [{epoch+1}/{train_cfg.epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')
        
        eval_results = evaluate(run_cfg, model, valid_loader, device, criterion, loss_function_name, desc=f"[Valid] [{epoch+1}/{train_cfg.epochs}]", class_names=class_names, log_class_metrics=True)

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
        
        # Warmup 기간이 끝난 후에만 원래 스케줄러를 사용합니다.
        if use_warmup and epoch == warmup_epochs - 1:
            logging.info(f"Warmup 종료. 에포크 {epoch + 2}부터 기존 스케줄러를 활성화합니다.")
            if scheduler:
                scheduler.step = original_scheduler_step # 원래 스케줄러 step 함수 복원
        
        if not (use_warmup and epoch < warmup_epochs) and scheduler:
            scheduler.step() # Warmup 기간이 아닐 때 스케줄러 step 호출

def inference(run_cfg, model_cfg, model, data_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=None):
    """저장된 모델로 추론 및 성능 평가를 수행합니다."""
    
    # --- ONNX 모델 직접 평가 분기 ---
    onnx_model_path = getattr(run_cfg, 'onnx_model_path', None)
    if onnx_model_path and os.path.exists(onnx_model_path):
        logging.info("="*50)
        logging.info(f"ONNX 모델 직접 평가를 시작합니다: '{onnx_model_path}'")
        if not onnxruntime:
            logging.error("ONNX Runtime이 설치되지 않았습니다. 'pip install onnxruntime'으로 설치해주세요.")
            return None
        try:
            onnx_session = onnxruntime.InferenceSession(onnx_model_path)
            dummy_input, _, _ = next(iter(data_loader))
            measure_onnx_performance(onnx_session, dummy_input)
            evaluate_onnx(run_cfg, onnx_session, data_loader, desc=f"[{mode_name} (ONNX)]", class_names=class_names, log_class_metrics=True)
        except Exception as e:
            logging.error(f"ONNX 모델 평가 중 오류 발생: {e}")
        return None # ONNX 직접 평가 후 종료

    logging.info(f"{mode_name} 모드를 시작합니다.")

    # --- [수정] Pruning된 모델과 일반 모델에 맞는 가중치 파일을 선택적으로 로드 ---
    # Pruning이 적용된 경우, 압축된 모델의 파라미터는 'pruned_model.pth'에 저장되어 있습니다.
    # main 함수에서 이미 압축된 모델 객체를 전달했으므로, 여기서는 가중치를 다시 로드할 필요가 없습니다.
    # 하지만, 만약 'inference' 모드로만 실행될 경우를 대비하여 로드 로직을 유지하되, 올바른 파일을 선택하도록 합니다.
    pruned_model_path = os.path.join(run_dir_path, 'pruned_model.pth')
    best_model_path = os.path.join(run_dir_path, run_cfg.model_path)

    # 'pruned_model.pth'가 존재하면 그것을 우선적으로 사용합니다.
    model_to_load = pruned_model_path if os.path.exists(pruned_model_path) else best_model_path

    # main 함수에서 이미 올바른 상태의 모델을 전달했으므로, 여기서는 가중치를 다시 로드할 필요가 없습니다.
    # 만약 'inference' 모드로만 실행될 경우, 아래 로직이 필요합니다.
    # if run_cfg.mode == 'inference':
    #     try:
    #         model.load_state_dict(torch.load(model_to_load, map_location=device))
    #         logging.info(f"'{model_to_load}' 가중치 로드 완료.")
    #     except Exception as e:
    #         logging.error(f"모델 가중치 로딩 중 오류 발생: {e}")
    #         return

    model.eval()

    # --- PyTorch 모델 성능 지표 측정 (FLOPS 및 더미 입력 생성) ---
    dummy_input = measure_model_flops(model, device, data_loader)
    single_dummy_input = dummy_input[0].unsqueeze(0) if dummy_input.shape[0] > 1 else dummy_input

    # --- 샘플 당 Forward Pass 시간 및 메모리 사용량 측정 ---
    avg_inference_time_per_sample = 0.0
    logging.info("GPU 캐시를 비우고, 단일 샘플에 대한 Forward Pass 시간 및 최대 GPU 메모리 사용량 측정을 시작합니다...")
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # 시간 측정을 위한 예열(warm-up)
        with torch.no_grad():
            for _ in range(10):
                _ = model(single_dummy_input)

        # 실제 시간 측정
        num_iterations = 100
        total_time = 0.0
        with torch.no_grad():
            for _ in range(num_iterations):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(single_dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                total_time += start_event.elapsed_time(end_event) # ms
        
        avg_inference_time_per_sample = total_time / num_iterations
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        logging.info(f"샘플 당 평균 Forward Pass 시간: {avg_inference_time_per_sample:.2f}ms (1개 샘플 x {num_iterations}회 반복)")
        logging.info(f"샘플 당 Forward Pass 시 최대 GPU 메모리 사용량: {peak_memory_mb:.2f} MB")
    else:
        logging.info("CUDA를 사용할 수 없어 CPU 추론 시간을 측정합니다.")

        # CPU 시간 측정을 위한 예열(warm-up)
        with torch.no_grad():
            for _ in range(10):
                _ = model(single_dummy_input)

        # 실제 시간 측정
        num_iterations = 100
        total_time = 0.0
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(single_dummy_input)
                end_time = time.time()
                total_time += (end_time - start_time) * 1000 # ms

        avg_inference_time_per_sample = total_time / num_iterations
        logging.info(f"샘플 당 평균 Forward Pass 시간 (CPU): {avg_inference_time_per_sample:.2f}ms (1개 샘플 x {num_iterations}회 반복)")
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
    
    # --- ONNX 변환 및 평가 ---
    evaluate_onnx_flag = getattr(run_cfg, 'evaluate_onnx', False)
    if evaluate_onnx_flag and onnxruntime and dummy_input is not None:
        logging.info("="*50)
        logging.info("ONNX 변환 및 평가를 시작합니다...")
        onnx_path = os.path.join(run_dir_path, f'model_{timestamp}.onnx')
        try:
            # 모델을 CPU로 이동하여 ONNX로 변환 (일반적으로 더 안정적)
            model.to('cpu')
            torch.onnx.export(model, dummy_input.to('cpu'), onnx_path,
                              export_params=True, opset_version=12,
                              do_constant_folding=True,
                              input_names=['input'], output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
            model.to(device) # 모델을 원래 장치로 복원
            logging.info(f"모델이 ONNX 형식으로 변환되어 '{onnx_path}'에 저장되었습니다.")

            # ONNX 런타임 세션 생성 및 평가
            onnx_session = onnxruntime.InferenceSession(onnx_path)
            measure_onnx_performance(onnx_session, dummy_input)
            evaluate_onnx(run_cfg, onnx_session, data_loader, desc=f"[{mode_name} (ONNX)]", class_names=class_names, log_class_metrics=True)

        except Exception as e:
            logging.error(f"ONNX 변환 또는 평가 중 오류 발생: {e}")

    return final_acc

def main():
    # =============================================================================
    # [최종 수정] NNI 모듈을 main 함수 내부에서 필요할 때 import 하도록 변경
    # =============================================================================
    from nni.compression.pruning import L1NormPruner, L2NormPruner, FPGMPruner
    from nni.compression.speedup import ModelSpeedup
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
    # 중첩된 scheduler_params 딕셔너리를 SimpleNamespace로 변환
    if hasattr(train_cfg, 'scheduler_params') and isinstance(train_cfg.scheduler_params, dict):
        train_cfg.scheduler_params = SimpleNamespace(**train_cfg.scheduler_params)
    if hasattr(train_cfg, 'warmup') and isinstance(train_cfg.warmup, dict):
        train_cfg.warmup = SimpleNamespace(**train_cfg.warmup)
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

    # =============================================================================
    # [최종 수정] timm 모델의 복합 레이어를 NNI가 이해할 수 있는 표준 레이어로 변환
    # =============================================================================
    # timm 라이브러리 버전 문제와 상관없이 직접 복합 레이어를 표준 레이어로 교체하는 함수
    def fuse_timm_norm_act_layers(module):
        """
        timm의 복합 레이어(예: BatchNormAct2d)를 표준 nn.Sequential(nn.BatchNorm2d, nn.ReLU)로 교체합니다.
        이 함수는 재귀적으로 모든 자식 모듈을 순회합니다.
        """
        # timm.layers.norm_act.BatchNormAct2d 타입을 동적으로 가져옵니다.
        try:
            from timm.layers import BatchNormAct2d
        except ImportError:
            # timm 버전이 다르거나 해당 모듈이 없는 경우, 함수를 조용히 종료합니다.
            return module

        for name, child in module.named_children():
            if isinstance(child, BatchNormAct2d):
                # [최종 수정] BatchNormAct2d는 nn.BatchNorm2d를 상속하므로, child 자체가 BN 레이어입니다.
                # BN 레이어에서 활성화 함수(act)만 분리하여 nn.Sequential로 재구성합니다.
                bn_layer = nn.BatchNorm2d(child.num_features, child.eps, child.momentum, child.affine, child.track_running_stats).to(next(child.parameters()).device)
                bn_layer.load_state_dict(child.state_dict())
                new_module = nn.Sequential(bn_layer, child.act)
                setattr(module, name, new_module)
                logging.info(f"  - 복합 레이어 '{name}'를 nn.Sequential(BatchNorm2d, ReLU)로 교체했습니다.")
            else:
                # 자식 모듈에 대해 재귀적으로 함수 호출
                fuse_timm_norm_act_layers(child)
        return module

    logging.info(f"timm 모델({baseline_model_name})의 복합 레이어를 표준 레이어로 변환합니다...")
    model = fuse_timm_norm_act_layers(model)

    log_model_parameters(model)
    pruner = None # pruner를 main 함수 스코프에서 정의

    # --- 경량화 적용 (config.yaml 설정에 따라) ---
    # 1. Pruning (가지치기) 적용
    if getattr(baseline_cfg, 'use_l1_pruning', False) and not getattr(baseline_cfg, 'use_l2_pruning', False) and not getattr(baseline_cfg, 'use_fpgm_pruning', False):
        logging.info("="*50)
        logging.info("L1 Norm Pruning을 시작합니다...")
        pruning_sparsity = getattr(baseline_cfg, 'pruning_sparsity', 0.5)
        
        # --- [수정] 마지막 분류 레이어를 제외한 모든 Conv2d와 Linear 레이어의 이름을 찾습니다. ---
        target_op_names = []
        last_linear_name = None
        for name, module in model.named_modules():
            # [최종 수정] 마지막 분류기와 직접 연결된 헤드 블록(conv_head, norm_head)을 Pruning 대상에서 제외합니다.
            # 이 레이어들이 Pruning되면 최종 분류기의 입력 차원과 불일치가 발생합니다.
            if name.startswith('classifier') or name.startswith('conv_head') or name.startswith('norm_head'):
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    logging.info(f"분류기 헤드 블록 '{name}'을(를) Pruning 대상에서 제외합니다.")
                continue

            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                target_op_names.append(name)

        # 이전 로직(마지막 linear 레이어 이름으로 제외)은 더 이상 필요하지 않습니다.
        # last_linear_name = 'classifier'
        if last_linear_name:
            logging.info(f"마지막 분류 레이어 '{last_linear_name}'를 Pruning 대상에서 제외합니다.")
            target_op_names.remove(last_linear_name)
        
        # op_names를 사용하여 Pruning할 레이어를 명시적으로 지정합니다.
        pruner_config_list = [{
            'op_names': target_op_names,
            'sparsity': pruning_sparsity
        }]
        
        logging.info(f"적용 희소도 (Sparsity): {pruning_sparsity}")
        
        # Pruner 생성 및 모델 압축
        pruner = L1NormPruner(model, pruner_config_list)
        model, masks = pruner.compress() # 모델에 마스크가 적용되고, masks를 반환받음
        
        logging.info("L1 Norm Pruning 적용 완료. 모델에 가지치기 마스크가 적용되었습니다.")
        logging.info("="*50)
    elif getattr(baseline_cfg, 'use_l2_pruning', False) and not getattr(baseline_cfg, 'use_fpgm_pruning', False):
        if getattr(baseline_cfg, 'use_l1_pruning', False) or getattr(baseline_cfg, 'use_fpgm_pruning', False):
            logging.warning("use_l1_pruning과 use_l2_pruning이 모두 true로 설정되었습니다. L2 Norm Pruning을 우선 적용합니다.")
        
        logging.info("="*50)
        logging.info("L2 Norm Pruning을 시작합니다...")
        pruning_sparsity = getattr(baseline_cfg, 'pruning_sparsity', 0.5)
        
        # --- [수정] 마지막 분류 레이어를 제외한 모든 Conv2d와 Linear 레이어의 이름을 찾습니다. ---
        target_op_names = []
        last_linear_name = None
        for name, module in model.named_modules():
            if name.startswith('classifier') or name.startswith('conv_head') or name.startswith('norm_head'):
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    logging.info(f"분류기 헤드 블록 '{name}'을(를) Pruning 대상에서 제외합니다.")
                continue

            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                target_op_names.append(name)

        # op_names를 사용하여 Pruning할 레이어를 명시적으로 지정합니다.
        pruner_config_list = [{
            'op_names': target_op_names,
            'sparsity': pruning_sparsity
        }]
        
        logging.info(f"적용 희소도 (Sparsity): {pruning_sparsity}")
        
        # Pruner 생성 및 모델 압축
        pruner = L2NormPruner(model, pruner_config_list)
        model, masks = pruner.compress() # 모델에 마스크가 적용되고, masks를 반환받음
        
        logging.info("L2 Norm Pruning 적용 완료. 모델에 가지치기 마스크가 적용되었습니다.")
        logging.info("="*50)
    elif getattr(baseline_cfg, 'use_fpgm_pruning', False):
        if getattr(baseline_cfg, 'use_l1_pruning', False) or getattr(baseline_cfg, 'use_l2_pruning', False):
            logging.warning("여러 Pruning 옵션이 활성화되었습니다. FPGM Pruning을 우선 적용합니다.")

        logging.info("="*50)
        logging.info("FPGM Pruning을 시작합니다...")
        pruning_sparsity = getattr(baseline_cfg, 'pruning_sparsity', 0.5)

        # --- [수정] 마지막 분류 레이어를 제외한 모든 Conv2d와 Linear 레이어의 이름을 찾습니다. ---
        target_op_names = []
        last_linear_name = None
        for name, module in model.named_modules():
            if name.startswith('classifier') or name.startswith('conv_head') or name.startswith('norm_head'):
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    logging.info(f"분류기 헤드 블록 '{name}'을(를) Pruning 대상에서 제외합니다.")
                continue

            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                target_op_names.append(name)

        # op_names를 사용하여 Pruning할 레이어를 명시적으로 지정합니다.
        pruner_config_list = [{
            'op_names': target_op_names,
            'sparsity': pruning_sparsity
        }]

        logging.info(f"적용 희소도 (Sparsity): {pruning_sparsity}")

        # Pruner 생성 및 모델 압축
        pruner = FPGMPruner(model, pruner_config_list)
        model, masks = pruner.compress() # 모델에 마스크가 적용되고, masks를 반환받음

        logging.info("FPGM Pruning 적용 완료. 모델에 가지치기 마스크가 적용되었습니다.")
        logging.info("="*50)

    # --- 옵티마이저 및 스케줄러 설정 ---
    optimizer, scheduler = None, None
    if run_cfg.mode == 'train':
        optimizer_name = getattr(train_cfg, 'optimizer', 'adamw').lower()
        logging.info("="*50)
        if optimizer_name == 'sgd':
            momentum = getattr(train_cfg, 'momentum', 0.9)
            weight_decay = getattr(train_cfg, 'weight_decay', 0.0001)
            logging.info(f"옵티마이저: SGD (lr={train_cfg.lr}, momentum={momentum}, weight_decay={weight_decay})")
            optimizer = optim.SGD(model.parameters(), lr=train_cfg.lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'nadam':
            weight_decay = getattr(train_cfg, 'weight_decay', 0.0)
            logging.info(f"옵티마이저: NAdam (lr={train_cfg.lr}, weight_decay={weight_decay})")
            optimizer = optim.NAdam(model.parameters(), lr=train_cfg.lr, weight_decay=weight_decay)
        elif optimizer_name == 'radam':
            weight_decay = getattr(train_cfg, 'weight_decay', 0.0)
            logging.info(f"옵티마이저: RAdam (lr={train_cfg.lr}, weight_decay={weight_decay})")
            optimizer = optim.RAdam(model.parameters(), lr=train_cfg.lr, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            weight_decay = getattr(train_cfg, 'weight_decay', 0.0)
            momentum = getattr(train_cfg, 'momentum', 0.0)
            logging.info(f"옵티마이저: RMSprop (lr={train_cfg.lr}, weight_decay={weight_decay}, momentum={momentum})")
            optimizer = optim.RMSprop(model.parameters(), lr=train_cfg.lr, weight_decay=weight_decay, momentum=momentum)
        else:
            weight_decay = getattr(train_cfg, 'weight_decay', 0.01)
            logging.info(f"옵티마이저: AdamW (lr={train_cfg.lr}, weight_decay={weight_decay})")
            optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=weight_decay)

        # scheduler_params가 없으면 빈 객체로 초기화
        scheduler_params = getattr(train_cfg, 'scheduler_params', SimpleNamespace())

        scheduler_name = getattr(train_cfg, 'scheduler', 'none').lower()
        if scheduler_name == 'multisteplr':
            milestones = getattr(train_cfg, 'milestones', [])
            gamma = getattr(train_cfg, 'gamma', 0.1)
            logging.info(f"스케줄러: MultiStepLR (milestones={milestones}, gamma={gamma})")
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name == 'cosineannealinglr':
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
        
        # --- [수정] 훈련 완료 후 Pruning 모델 Export 및 저장 ---
        if pruner:
            logging.info("="*50)
            logging.info("Pruning된 모델을 Export합니다 (가중치를 영구적으로 제거).")
            
            # 1. 훈련 중 가장 성능이 좋았던 모델의 가중치와 마스크 정보를 불러옵니다.
            model.load_state_dict(torch.load(os.path.join(run_dir_path, run_cfg.model_path), map_location=device))
            
            # 2. 모델을 감싸고 있던 Wrapper를 제거하여 순수한 Pytorch 모델로 되돌립니다.
            pruner.unwrap_model()
            logging.info("Pruner.unwrap_model() 완료. 모델에서 Wrapper가 제거되었습니다.")

            # 3. ModelSpeedup을 사용하여 마스크를 기반으로 모델을 물리적으로 가속(압축)합니다.
            #    이를 위해 모델에 입력될 더미 데이터가 필요합니다.
            dummy_input = torch.randn(1, 3, 224, 224).to(device) # config의 img_size를 사용하면 더 좋습니다.

            # [최종 수정] NNI의 기본 추적 기능 대신, 안정적인 torch.fx.symbolic_trace를 사용하여
            # 모델의 연산 그래프(GraphModule)를 명시적으로 생성합니다.
            # 이렇게 생성된 그래프를 ModelSpeedup에 전달하면 추적 오류를 방지할 수 있습니다.
            from torch.fx import symbolic_trace
            # [최종 수정] symbolic_trace와 ModelSpeedup은 eval() 모드에서 실행해야 합니다.
            # train() 모드에서 배치 크기가 1인 dummy_input을 사용하면 BatchNorm 레이어에서 오류가 발생합니다.
            model.eval()
            graph_module = symbolic_trace(model)

            # [최종 수정] nn.Identity 모듈을 위한 사용자 정의 Replacer 클래스 정의
            # ModelSpeedup이 Identity 레이어를 만났을 때 처리 방법을 몰라 경고를 발생시키고
            # 채널 정보 전파에 실패하는 문제를 해결합니다.
            from nni.compression.speedup.replacer import Replacer
            class IdentityReplacer(Replacer):
                def replace_modules(self, speedup):
                    for node in speedup.graph_module.graph.nodes:
                        if node.op == 'call_module':
                            module = speedup.fetch_attr(node.target)
                            if isinstance(module, nn.Identity):
                                # Identity 모듈은 아무것도 변경하지 않으므로, 교체되었다고만 표시합니다.
                                speedup.node_infos[node].replaced = True

            # 생성된 그래프(graph_module)를 사용하여 ModelSpeedup을 초기화합니다.
            speedup = ModelSpeedup(model, dummy_input, masks, graph_module=graph_module, customized_replacers=[IdentityReplacer()])
            model = speedup.speedup_model()
            logging.info("ModelSpeedup 완료. 모델 구조가 영구적으로 변경 및 압축되었습니다.")

            # 4. 압축된 모델의 state_dict를 새로운 파일로 저장합니다.
            pruned_model_path = os.path.join(run_dir_path, 'pruned_model.pth')
            torch.save(model.state_dict(), pruned_model_path)
            logging.info(f"Pruning이 적용된 모델이 '{pruned_model_path}'에 저장되었습니다.")
            
            # 5. 압축된 모델의 파라미터 수를 다시 확인합니다.
            # 이 시점에서 파라미터 수가 실제로 줄어든 것을 확인할 수 있습니다.
            logging.info("Export된 Pruned 모델의 파라미터 수를 다시 확인합니다.")
            log_model_parameters(model)

        logging.info("="*50)
        logging.info("훈련 완료. 최종 모델 성능을 테스트 세트로 평가합니다.")
        final_acc = inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Test", class_names=class_names)

        if final_acc is not None:
            log_filename = f"log_{timestamp}.log"
            log_file_path = os.path.join(run_dir_path, log_filename)
            plot_and_save_val_accuracy_graph(log_file_path, run_dir_path, final_acc, timestamp)
            plot_and_save_train_val_accuracy_graph(log_file_path, run_dir_path, final_acc, timestamp)
            plot_and_save_f1_normal_graph(log_file_path, run_dir_path, timestamp, class_names)
            plot_and_save_loss_graph(log_file_path, run_dir_path, timestamp)
            plot_and_save_lr_graph(log_file_path, run_dir_path, timestamp)
            plot_and_save_compiled_graph(run_dir_path, timestamp)

    elif run_cfg.mode == 'inference':
        # onnx_model_path가 지정된 경우, model 객체는 필요 없으므로 None을 전달합니다.
        onnx_model_path = getattr(run_cfg, 'onnx_model_path', None)
        if onnx_model_path and os.path.exists(onnx_model_path):
            logging.info(f"'{onnx_model_path}' ONNX 파일 평가를 위해 PyTorch 모델 생성을 건너뜁니다.")
            inference(run_cfg, model_cfg, None, test_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=class_names)
        else:
            log_model_parameters(model)
            inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=class_names)


if __name__ == '__main__':
    main()