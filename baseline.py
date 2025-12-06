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
import copy
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
        # [최종 수정] ONNX 변환 오류의 근본 원인인 AdaptiveAvgPool2d 업샘플링을
        # ONNX가 지원하는 Upsample 레이어로 명시적으로 교체합니다.
        # self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.avgpool = nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False)
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

def train(run_cfg, train_cfg, baseline_cfg, config, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names, pos_weight, epoch_offset=0):
    """모델 훈련 및 검증을 수행하고 최고 성능 모델을 저장합니다."""
    logging.info("train 모드를 시작합니다.")
    model_path = os.path.join(run_dir_path, run_cfg.pth_best_name)

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
    is_best_saved = False # 베스트 모델이 저장되었는지 확인하는 플래그

    # --- Warmup 설정 ---
    warmup_cfg = getattr(train_cfg, 'warmup', None)
    use_warmup = warmup_cfg and getattr(warmup_cfg, 'enabled', False)
    if use_warmup:
        warmup_epochs = getattr(warmup_cfg, 'epochs', 0)
        warmup_start_lr = getattr(warmup_cfg, 'start_lr', 0.0)
        warmup_end_lr = train_cfg.lr # Warmup 종료 LR은 메인 LR로 설정
        logging.info(f"Warmup 활성화: {warmup_epochs} 에포크 동안 LR을 {warmup_start_lr}에서 {warmup_end_lr}로 선형 증가시킵니다.")
    else: # Warmup을 사용하지 않을 경우 관련 변수 초기화
        warmup_epochs = 0

        # Warmup 기간 동안에는 스케줄러를 비활성화합니다.
    original_scheduler_step = scheduler.step if scheduler else lambda: None
    if scheduler:
        scheduler.step = lambda: None # type: ignore

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
        logging.info(f"[LR]    [{epoch + 1 + epoch_offset}/{train_cfg.epochs + epoch_offset}] | Learning Rate: {current_lr:.6f}")

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1 + epoch_offset}/{train_cfg.epochs + epoch_offset} [Training]", leave=False, disable=not getattr(run_cfg, 'show_log', True))
        for images, labels, _ in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = model(images)
            if loss_function_name == 'bcewithlogitsloss':
                loss = criterion(outputs[:, 1].unsqueeze(1), labels.float().unsqueeze(1))
            else: # crossentropyloss
                loss = criterion(outputs, labels)
            loss.backward()

            # --- [추가] Network Slimming을 위한 L1 정규화 손실 추가 ---
            is_slimming_pretrain = getattr(baseline_cfg, 'use_slimming_pruning', False) and \
                                   train_cfg.epochs == config.get('training_baseline', {}).get('epochs')

            if is_slimming_pretrain:
                l1_loss = torch.tensor(0., device=device)
                slimming_l1_strength = 1e-4
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        l1_loss += torch.norm(module.weight, 1)
                loss += slimming_l1_strength * l1_loss

            optimizer.step()

            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            running_loss += loss_val

            _, predicted = torch.max(outputs.data, 1) # outputs는 로짓이므로 그대로 사용
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            step_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{step_lr:.6f}")

        train_acc = 100 * correct / total
        logging.info(f'[Train] [{epoch + 1 + epoch_offset}/{train_cfg.epochs + epoch_offset}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')
        
        eval_results = evaluate(run_cfg, model, valid_loader, device, criterion, loss_function_name, desc=f"[Valid] [{epoch + 1 + epoch_offset}/{train_cfg.epochs + epoch_offset}]", class_names=class_names, log_class_metrics=True)

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
        
        # 최고 성능 모델 저장
        if is_best:
            best_metric = current_metric
            torch.save(model.state_dict(), model_path)
            criterion_name = best_model_criterion.replace('_', ' ')
            is_best_saved = True
            logging.info(f"[Best Model Saved] ({criterion_name}: {best_metric:.4f}) -> '{model_path}'")
        
        # Warmup 기간이 끝난 후에만 원래 스케줄러를 사용합니다.
        if use_warmup and epoch == warmup_epochs - 1:
            logging.info(f"Warmup 종료. 에포크 {epoch + 2}부터 기존 스케줄러를 활성화합니다.")
            if scheduler:
                scheduler.step = original_scheduler_step # 원래 스케줄러 step 함수 복원
        
        if not (use_warmup and epoch < warmup_epochs) and scheduler:
            scheduler.step() # Warmup 기간이 아닐 때 스케줄러 step 호출
    
    # 만약 훈련 동안 한 번도 best model이 저장되지 않았다면(e.g., loss가 계속 nan), 마지막 모델이라도 저장합니다.
    if not is_best_saved:
        torch.save(model.state_dict(), model_path)
        logging.warning(f"훈련 동안 성능 개선이 없어 Best 모델이 저장되지 않았습니다. 마지막 에포크의 모델을 '{model_path}'에 저장합니다.")


def inference(run_cfg, model_cfg, model, data_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=None):
    """저장된 모델로 추론 및 성능 평가를 수행합니다."""
    
    # --- ONNX 모델 직접 평가 분기 ---
    onnx_inference_path = getattr(run_cfg, 'onnx_inference_path', None)
    if onnx_inference_path and os.path.exists(onnx_inference_path):
        logging.info("="*50)
        logging.info(f"ONNX 모델 직접 평가를 시작합니다: '{onnx_inference_path}'")
        if not onnxruntime:
            logging.error("ONNX Runtime이 설치되지 않았습니다. 'pip install onnxruntime'으로 설치해주세요.")
            return None
        try:
            logging.info(f"ONNX Runtime (v{onnxruntime.__version__})으로 평가를 시작합니다.")
            onnx_session = onnxruntime.InferenceSession(onnx_inference_path)
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
    best_model_path = os.path.join(run_dir_path, run_cfg.pth_best_name)

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
        iteration_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(single_dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                iteration_times.append(start_event.elapsed_time(end_event)) # ms
        
        avg_inference_time_per_sample = np.mean(iteration_times)
        std_inference_time_per_sample = np.std(iteration_times)
        fps = 1000 / avg_inference_time_per_sample if avg_inference_time_per_sample > 0 else 0
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        logging.info(f"샘플 당 평균 Forward Pass 시간: {avg_inference_time_per_sample:.2f}ms (std: {std_inference_time_per_sample:.2f}ms), FPS: {fps:.2f} (1개 샘플 x {num_iterations}회 반복)")
        logging.info(f"샘플 당 Forward Pass 시 최대 GPU 메모리 사용량: {peak_memory_mb:.2f} MB")
    else:
        logging.info("CUDA를 사용할 수 없어 CPU 추론 시간을 측정합니다.")

        # CPU 시간 측정을 위한 예열(warm-up)
        with torch.no_grad():
            for _ in range(10):
                _ = model(single_dummy_input)

        # 실제 시간 측정
        num_iterations = 100
        iteration_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(single_dummy_input)
                end_time = time.time()
                iteration_times.append((end_time - start_time) * 1000) # ms

        avg_inference_time_per_sample = np.mean(iteration_times)
        std_inference_time_per_sample = np.std(iteration_times)
        fps = 1000 / avg_inference_time_per_sample if avg_inference_time_per_sample > 0 else 0
        logging.info(f"샘플 당 평균 Forward Pass 시간 (CPU): {avg_inference_time_per_sample:.2f}ms (std: {std_inference_time_per_sample:.2f}ms), FPS: {fps:.2f} (1개 샘플 x {num_iterations}회 반복)")
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
            # --- ONNX 런타임 세션 옵션 설정 ---
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            torch.onnx.export(model, dummy_input.to('cpu'), onnx_path,
                              export_params=True, opset_version=12,
                              do_constant_folding=True,
                              input_names=['input'], output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
            model.to(device) # 모델을 원래 장치로 복원
            logging.info(f"모델이 ONNX 형식으로 변환되어 '{onnx_path}'에 저장되었습니다.")

            # ONNX 런타임 세션 생성 및 평가
            onnx_session = onnxruntime.InferenceSession(onnx_path, sess_options=sess_options)
            measure_onnx_performance(onnx_session, dummy_input)
            evaluate_onnx(run_cfg, onnx_session, data_loader, desc=f"[{mode_name} (ONNX)]", class_names=class_names, log_class_metrics=True)

        except Exception as e:
            logging.error(f"ONNX 변환 또는 평가 중 오류 발생: {e}")

    return final_acc

def main():
    # =============================================================================
    # [해결] timm의 복합 레이어를 분리하기 위해 타입을 import합니다.
    try:
        from timm.layers import BatchNormAct2d
        # [추가] torch-pruning 라이브러리를 main 함수 내부에서 import 합니다.
        import torch_pruning as tp
    except ImportError:
        BatchNormAct2d = None
        tp = None
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
    finetune_cfg = SimpleNamespace(**config.get('finetuning_pruned', {})) # Fine-tuning 설정 로드
    # 중첩된 scheduler_params 딕셔너리를 SimpleNamespace로 변환
    if hasattr(train_cfg, 'scheduler_params') and isinstance(train_cfg.scheduler_params, dict):
        train_cfg.scheduler_params = SimpleNamespace(**train_cfg.scheduler_params)
    if hasattr(train_cfg, 'warmup') and isinstance(train_cfg.warmup, dict):
        train_cfg.warmup = SimpleNamespace(**train_cfg.warmup)
    run_cfg.dataset = SimpleNamespace(**run_cfg.dataset)
    
    # Fine-tuning 설정도 동일하게 변환
    if hasattr(finetune_cfg, 'scheduler_params') and isinstance(finetune_cfg.scheduler_params, dict):
        finetune_cfg.scheduler_params = SimpleNamespace(**finetune_cfg.scheduler_params)
    if hasattr(finetune_cfg, 'warmup') and isinstance(finetune_cfg.warmup, dict):
        finetune_cfg.warmup = SimpleNamespace(**finetune_cfg.warmup)
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
        # ONNX 직접 추론 경로가 설정되었는지 확인
        onnx_inference_path = getattr(run_cfg, 'onnx_inference_path', None)
        if not (onnx_inference_path and os.path.exists(onnx_inference_path)):
            # ONNX 경로가 없는 경우에만 pth_inference_dir 경로를 검사합니다.
            run_dir_path = getattr(run_cfg, 'pth_inference_dir', None)
            if getattr(run_cfg, 'show_log', True) and (not run_dir_path or not os.path.isdir(run_dir_path)):
                logging.error("추론 모드에서는 'config.yaml'에 'pth_inference_dir'를 올바르게 설정해야 합니다.")
                exit()
        else:
            # ONNX 경로가 있으면 pth_inference_dir은 무시하고 현재 디렉토리를 임시로 사용합니다.
            run_dir_path = '.'
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
        if BatchNormAct2d is None:
            return module

        for name, child in module.named_children():
            if isinstance(child, BatchNormAct2d):
                # BatchNormAct2d는 nn.BatchNorm2d를 상속합니다.
                # 새로운 nn.BatchNorm2d를 만들고 state_dict를 복사합니다.
                bn = nn.BatchNorm2d(child.num_features, child.eps, child.momentum, child.affine, child.track_running_stats)
                bn.load_state_dict(child.state_dict())
                # 활성화 함수(act)와 결합하여 nn.Sequential로 만듭니다.
                # timm 모델은 device 정보가 없을 수 있으므로, to(device)를 추가합니다.
                new_module = nn.Sequential(bn, child.act).to(device)
                setattr(module, name, new_module)
                logging.info(f"  - 복합 레이어 '{name}'를 nn.Sequential(BatchNorm2d, Activation)으로 교체했습니다.")
            else:
                # 자식 모듈에 대해 재귀적으로 함수 호출
                fuse_timm_norm_act_layers(child)
        return module

    logging.info(f"timm 모델({baseline_model_name})의 복합 레이어를 표준 레이어로 변환합니다...")
    model = fuse_timm_norm_act_layers(model)

    log_model_parameters(model)
    pruner = None # pruner를 main 함수 스코프에서 정의
    masks = None  # masks를 main 함수 스코프에서 정의
    # --- Pruning 적용 여부 확인 ---
    use_pruning = getattr(baseline_cfg, 'use_l1_pruning', False) or \
                  getattr(baseline_cfg, 'use_l2_pruning', False) or \
                  getattr(baseline_cfg, 'use_fpgm_pruning', False) or \
                  getattr(baseline_cfg, 'use_lamp_pruning', False) or \
                  getattr(baseline_cfg, 'use_depgraph_pruning', False) or \
                  getattr(baseline_cfg, 'use_taylor_pruning', False) or \
                  getattr(baseline_cfg, 'use_slimming_pruning', False)

    def run_torch_pruning(model, baseline_cfg, model_cfg, device, train_loader=None, criterion=None):
        """torch-pruning 라이브러리를 사용하여 DepGraph 기반 Pruning을 수행합니다."""
        if tp is None:
            logging.error("DepGraph Pruning을 사용하려면 'torch-pruning' 라이브러리를 설치해야 합니다. (pip install torch-pruning)")
            return model
        
        pruning_sparsity = getattr(baseline_cfg, 'pruning_sparsity', 0.5)
        dummy_input = torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device)

        # --- Pruning 전략 선택 ---
        pruner_class = None
        pruner_kwargs = {}
        if getattr(baseline_cfg, 'use_l1_pruning', False):
            logging.info("torch-pruning을 사용한 L1 Norm Pruning을 시작합니다...")
            imp = tp.importance.MagnitudeImportance(p=1)
            pruner_class = tp.pruner.MagnitudePruner
            pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
        elif getattr(baseline_cfg, 'use_l2_pruning', False):
            logging.info("torch-pruning을 사용한 L2 Norm Pruning을 시작합니다...")
            imp = tp.importance.MagnitudeImportance(p=2)
            pruner_class = tp.pruner.MagnitudePruner
            pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
        elif getattr(baseline_cfg, 'use_depgraph_pruning', False):
            logging.info("DepGraph Pruning (Magnitude-based)을 시작합니다...")
            imp = tp.importance.MagnitudeImportance(p=1)
            pruner_class = tp.pruner.MagnitudePruner
            pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
        elif getattr(baseline_cfg, 'use_lamp_pruning', False):
            logging.info("LAMP Pruning을 시작합니다...")
            # LAMP는 중요도 계산에 더 많은 반복이 필요할 수 있습니다.
            imp = tp.importance.LAMPImportance(p=2, group_reduction="mean")
            pruner_class = tp.pruner.MagnitudePruner
            pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
        elif getattr(baseline_cfg, 'use_slimming_pruning', False):
            logging.info("Network Slimming (BN-Scale) Pruning을 시작합니다...")
            imp = tp.importance.BNScaleImportance()
            pruner_class = tp.pruner.BNScalePruner
            pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
        elif getattr(baseline_cfg, 'use_taylor_pruning', False):
            logging.info("Group Taylor Pruning을 시작합니다...")
            # Taylor Pruning은 그래디언트 기반 중요도를 사용합니다.
            imp = tp.importance.TaylorImportance()
            # 중요도 계산을 위해 그래디언트가 필요하므로, 데이터로더에서 샘플 배치를 가져와 forward/backward pass를 수행합니다.
            if train_loader is None or criterion is None:
                logging.error("Taylor Pruning을 사용하려면 train_loader와 criterion이 필요합니다.")
                return model
            
            logging.info("Taylor 중요도 계산을 위해 샘플 배치에 대한 그래디언트를 계산합니다...")
            images, labels, _ = next(iter(train_loader))
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()

            pruner_class = tp.pruner.MagnitudePruner
            pruner_kwargs = {'importance': imp, 'pruning_ratio': pruning_sparsity}
        else:
            logging.warning("활성화된 torch-pruning 기법이 없습니다.")
            return model

        # --- 무시할 레이어(Ignored Layers) 설정 ---
        ignored_layers = []
        last_layer = None
        if hasattr(model, 'fc'):
            last_layer = model.fc
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            last_layer = model.classifier[-1]
        elif hasattr(model, 'classifier'):
            last_layer = model.classifier
        elif hasattr(model, 'head'):
            last_layer = model.head
        
        if last_layer:
            ignored_layers.append(last_layer)
            logging.info(f"분류 레이어 '{type(last_layer).__name__}'을(를) Pruning 대상에서 제외합니다.")

        # --- Pruner 생성 및 실행 ---
        pruner = pruner_class(
            model,
            dummy_input,
            ignored_layers=ignored_layers,
            **pruner_kwargs
        )

        # Taylor Pruning의 경우, step()에서 그래디언트를 사용합니다.
        if getattr(baseline_cfg, 'use_taylor_pruning', False):
            pruner.step(interactive=False)
        else:
            pruner.step()
        logging.info(f"torch-pruning 완료. 모델 구조가 희소도({pruning_sparsity})에 맞춰 변경되었습니다.")
        logging.info("="*50)
        return model

    def find_sparsity_for_target_flops(model, baseline_cfg, model_cfg, device, train_loader, criterion):
        """이진 탐색을 사용하여 목표 FLOPs에 가장 가까운 pruning_sparsity를 찾습니다."""
        target_gflops = getattr(baseline_cfg, 'pruning_flops_target', 0.0)
        if target_gflops <= 0:
            return getattr(baseline_cfg, 'pruning_sparsity', 0.5)

        logging.info("="*80)
        logging.info(f"목표 FLOPs ({target_gflops:.4f} GFLOPs)에 맞는 최적의 Pruning 희소도를 탐색합니다...")

        # 원본 FLOPs 측정
        original_macs, _ = profile(model, inputs=(torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device),), verbose=False)
        original_gflops = original_macs * 2 / 1e9
        logging.info(f"원본 모델 FLOPs: {original_gflops:.4f} GFLOPs")


        # 이진 탐색을 위한 초기값 설정
        low_sparsity, high_sparsity = 0.0, 0.99
        best_sparsity = 0.0
        min_flops_diff = float('inf')
        
        # 원본 모델을 복사하여 탐색 과정에서 원본이 변경되지 않도록 함
        original_model = copy.deepcopy(model)
        dummy_input = torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device)

        # 이진 탐색 반복 (20회 정도면 충분한 정밀도 확보 가능)
        for i in range(20):
            current_sparsity = (low_sparsity + high_sparsity) / 2
            
            # 임시 모델에 현재 희소도로 Pruning 적용
            temp_model = copy.deepcopy(original_model)
            
            # run_torch_pruning과 동일한 로직으로 Pruning을 시뮬레이션
            # 단, 실제 모델을 변경하는 대신 FLOPs만 계산
            temp_baseline_cfg = copy.deepcopy(baseline_cfg)
            temp_baseline_cfg.pruning_sparsity = current_sparsity
            
            try:
                pruned_temp_model = run_torch_pruning(temp_model, temp_baseline_cfg, model_cfg, device, train_loader=train_loader, criterion=criterion)
                
                # Pruning된 임시 모델의 FLOPs 계산
                macs, _ = profile(pruned_temp_model, inputs=(dummy_input,), verbose=False)
                current_gflops = macs * 2 / 1e9

                flops_diff = abs(current_gflops - target_gflops)
                reduction_ratio = (1 - current_gflops / original_gflops) * 100
                logging.info(f"  [탐색 {i+1:2d}] 희소도: {current_sparsity:.4f} -> FLOPs: {current_gflops:.4f} GFLOPs (감소율: {reduction_ratio:.2f}%)")

                # 최고 기록 갱신
                if flops_diff < min_flops_diff:
                    min_flops_diff = flops_diff
                    best_sparsity = current_sparsity

                # 이진 탐색 범위 조정
                if current_gflops > target_gflops: # FLOPs가 목표보다 크면, 희소도를 높여야 함 (더 많이 제거)
                    low_sparsity = current_sparsity
                else: # FLOPs가 목표보다 작거나 같으면, 희소도를 낮춰야 함 (덜 제거)
                    high_sparsity = current_sparsity
            except Exception as e:
                logging.warning(f"  [탐색 {i+1:2d}] 희소도 {current_sparsity:.4f} 적용 중 오류 발생: {e}. 이 희소도는 건너뜁니다.")
                low_sparsity = current_sparsity # 오류 발생 시 해당 구간 탐색 회피

        logging.info(f"탐색 완료. 목표 FLOPs({target_gflops:.4f})에 가장 근접한 최적 희소도는 {best_sparsity:.4f} 입니다.")
        logging.info("="*80)
        return best_sparsity

    # --- 옵티마이저 및 스케줄러 생성 함수 ---
    def create_optimizer_and_scheduler(cfg, model):
        optimizer_name = getattr(cfg, 'optimizer', 'adamw').lower()
        if optimizer_name == 'sgd':
            momentum = getattr(cfg, 'momentum', 0.9)
            weight_decay = getattr(cfg, 'weight_decay', 0.0001)
            logging.info(f"옵티마이저: SGD (lr={cfg.lr}, momentum={momentum}, weight_decay={weight_decay})")
            optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=momentum, weight_decay=weight_decay)
        # ... (다른 옵티마이저들 추가) ...
        else: # adamw
            weight_decay = getattr(cfg, 'weight_decay', 0.01)
            logging.info(f"옵티마이저: AdamW (lr={cfg.lr}, weight_decay={weight_decay})")
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)

        scheduler_params = getattr(cfg, 'scheduler_params', SimpleNamespace())
        scheduler_name = getattr(cfg, 'scheduler', 'none').lower()
        scheduler = None
        if scheduler_name == 'multisteplr':
            milestones = getattr(cfg, 'milestones', [])
            gamma = getattr(cfg, 'gamma', 0.1)
            logging.info(f"스케줄러: MultiStepLR (milestones={milestones}, gamma={gamma})")
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name == 'cosineannealinglr':
            T_max = getattr(scheduler_params, 'T_max', cfg.epochs)
            eta_min = getattr(scheduler_params, 'eta_min', 0.0)
            logging.info(f"스케줄러: CosineAnnealingLR (T_max={T_max}, eta_min={eta_min})")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        else:
            logging.info("스케줄러를 사용하지 않습니다.")
        
        return optimizer, scheduler

    # --- 모드에 따라 실행 ---
    if run_cfg.mode == 'train':
        # --- 1단계: 사전 훈련 (Pruning 없이) ---
        logging.info("="*80)
        logging.info("단계 1/2: 사전 훈련(Pre-training)을 시작합니다...")
        logging.info("="*80)
        
        optimizer, scheduler = create_optimizer_and_scheduler(train_cfg, model)
        train(run_cfg, train_cfg, baseline_cfg, config, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names, pos_weight)
        
        # --- 2단계: Pruning 및 미세 조정 (Fine-tuning) ---
        if use_pruning:
            logging.info("="*80)
            logging.info("단계 2/2: Pruning 및 미세 조정(Fine-tuning)을 시작합니다...")
            logging.info("="*80)

            # 사전 훈련된 최고 성능 모델 불러오기
            best_model_path = os.path.join(run_dir_path, run_cfg.pth_best_name)
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            logging.info(f"사전 훈련된 모델 '{best_model_path}'을(를) 불러왔습니다.")

            # 목표 FLOPs가 설정된 경우, 최적의 희소도를 자동으로 계산
            if getattr(baseline_cfg, 'pruning_flops_target', 0.0) > 0:
                # 이진 탐색으로 최적의 희소도 찾기
                # Taylor Pruning은 criterion이 필요하므로 임시로 생성
                loss_function_name = getattr(train_cfg, 'loss_function', 'CrossEntropyLoss').lower()
                criterion = nn.CrossEntropyLoss() if loss_function_name != 'bcewithlogitsloss' else nn.BCEWithLogitsLoss()
                optimal_sparsity = find_sparsity_for_target_flops(model, baseline_cfg, model_cfg, device, train_loader, criterion)
                # 찾은 희소도를 설정에 반영
                baseline_cfg.pruning_sparsity = optimal_sparsity

            # Pruning 적용
            # [수정] L1, L2 Pruning도 torch-pruning으로 통합
            use_torch_pruning = getattr(baseline_cfg, 'use_l1_pruning', False) or \
                                getattr(baseline_cfg, 'use_l2_pruning', False) or \
                                getattr(baseline_cfg, 'use_depgraph_pruning', False) or \
                                getattr(baseline_cfg, 'use_fpgm_pruning', False) or \
                                getattr(baseline_cfg, 'use_fpgm_pruning', False) or \
                                getattr(baseline_cfg, 'use_taylor_pruning', False) or \
                                getattr(baseline_cfg, 'use_lamp_pruning', False) or \
                                getattr(baseline_cfg, 'use_slimming_pruning', False)

            # Pruning 전 원본 모델의 FLOPs 측정
            original_macs, _ = profile(model, inputs=(torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device),), verbose=False)
            original_gflops = original_macs * 2 / 1e9
            if use_torch_pruning:
                # torch-pruning 계열의 기법은 모델을 직접 수정합니다.
                # Taylor Pruning은 그래디언트 계산을 위해 추가 정보가 필요합니다.
                if getattr(baseline_cfg, 'use_taylor_pruning', False):
                    # 임시 손실 함수 생성
                    loss_function_name = getattr(train_cfg, 'loss_function', 'CrossEntropyLoss').lower()
                    criterion = nn.CrossEntropyLoss() if loss_function_name != 'bcewithlogitsloss' else nn.BCEWithLogitsLoss()
                    model = run_torch_pruning(model, baseline_cfg, model_cfg, device, train_loader=train_loader, criterion=criterion)
                else:
                    model = run_torch_pruning(model, baseline_cfg, model_cfg, device)

                log_model_parameters(model) # Pruning 후 파라미터 수 확인

                # Pruning 후 FLOPs 측정 및 감소율 로깅
                pruned_macs, _ = profile(model, inputs=(torch.randn(1, 3, model_cfg.img_size, model_cfg.img_size).to(device),), verbose=False)
                pruned_gflops = pruned_macs * 2 / 1e9
                flops_reduction_ratio = (1 - pruned_gflops / original_gflops) * 100
                logging.info(f"FLOPs가 {original_gflops:.4f} GFLOPs에서 {pruned_gflops:.4f} GFLOPs로 감소했습니다 (감소율: {flops_reduction_ratio:.2f}%).")

                pruner = None # torch-pruning은 NNI pruner 객체를 사용하지 않음.
            
            if use_torch_pruning: # [수정] torch-pruning이 사용된 경우에만 미세조정 실행
                # 미세 조정을 위한 새로운 옵티마이저 및 스케줄러 생성
                logging.info("미세 조정을 위한 새로운 옵티마이저와 스케줄러를 생성합니다.")
                finetune_optimizer, finetune_scheduler = create_optimizer_and_scheduler(finetune_cfg, model)
                
                # 미세 조정 훈련 실행
                train(run_cfg, finetune_cfg, baseline_cfg, config, model, finetune_optimizer, finetune_scheduler, train_loader, valid_loader, device, run_dir_path, class_names, pos_weight, epoch_offset=train_cfg.epochs)
            else:
                logging.info("활성화된 Pruning 방법이 없어 미세 조정을 건너뜁니다.")

        # --- 최종 단계: Pruning된 모델 저장 및 평가 ---
        # torch-pruning은 모델을 직접 수정하므로, 미세조정 후 최종 모델을 저장합니다.
        if use_pruning:
            pruned_model_path = os.path.join(run_dir_path, 'pruned_model.pth')
            torch.save(model.state_dict(), pruned_model_path)
            logging.info(f"Pruning이 적용된 모델이 '{pruned_model_path}'에 저장되었습니다.")

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
        # onnx_inference_path가 지정된 경우, model 객체는 필요 없으므로 None을 전달합니다.
        onnx_inference_path = getattr(run_cfg, 'onnx_inference_path', None)
        if onnx_inference_path and os.path.exists(onnx_inference_path):
            logging.info(f"'{onnx_inference_path}' ONNX 파일 평가를 위해 PyTorch 모델 생성을 건너뜁니다.")
            inference(run_cfg, model_cfg, None, test_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=class_names)
        else:
            log_model_parameters(model)
            inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=class_names)


if __name__ == '__main__':
    main()
