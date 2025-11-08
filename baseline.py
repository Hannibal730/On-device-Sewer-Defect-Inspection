import os
from tqdm import tqdm
import torch
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
import time
from dataloader import prepare_data # 데이터 로딩 함수 임포트
import schedulefree

try:
    from thop import profile
except ImportError:
    profile = None

from plot import plot_and_save_train_val_accuracy_graph, plot_and_save_val_accuracy_graph, plot_and_save_confusion_matrix

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
    run_dir_name = f"run_{baseline_model_name}_{timestamp}"
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

def create_baseline_model(model_name, num_labels, pretrained=True):
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
def evaluate(run_cfg, model, data_loader, device, desc="Evaluating", class_names=None, log_class_metrics=False):
    """모델을 평가하고 정확도, 정밀도, 재현율, F1 점수를 로깅합니다."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    show_log = getattr(run_cfg, 'show_log', True)
    progress_bar = tqdm(data_loader, desc=desc, leave=False, disable=not show_log)
    with torch.no_grad():
        for images, labels, _ in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if total == 0:
        logging.warning("평가 데이터가 없습니다. 평가를 건너뜁니다.")
        return {'accuracy': 0.0, 'f1': 0.0, 'labels': [], 'preds': []}

    accuracy = 100 * correct / total
    
    if desc.startswith("[Valid]"):
        acc_label = "Val Acc"
        log_message = f'{desc} | {acc_label}: {accuracy:.2f}%'
    else:
        acc_label = "Test Acc"
        log_message = f'{desc} {acc_label}: {accuracy:.2f}%'
    logging.info(log_message)

    if log_class_metrics and class_names:
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        logging.info("-" * 30)
        for i, class_name in enumerate(class_names):
            log_line = (f"  - Metrics for '{class_name}': "
                        f"Precision: {precision_per_class[i]:.4f} | "
                        f"Recall: {recall_per_class[i]:.4f} | "
                        f"F1: {f1_per_class[i]:.4f}")
            logging.info(log_line)
        logging.info("-" * 30)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_per_class': f1_per_class if log_class_metrics and class_names else None,
        'labels': all_labels,
        'preds': all_preds
    }

def train(run_cfg, train_cfg, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names):
    """모델 훈련 및 검증을 수행하고 최고 성능 모델을 저장합니다."""
    logging.info("train 모드를 시작합니다.")
    model_path = os.path.join(run_dir_path, run_cfg.model_path)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0
    best_model_criterion = getattr(train_cfg, 'best_model_criterion', 'F1_average')

    for epoch in range(train_cfg.epochs):
        logging.info("-" * 50)
        model.train()
        if optimizer and hasattr(optimizer, 'train'):
            optimizer.train()

        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.epochs} [Training]", leave=False, disable=not getattr(run_cfg, 'show_log', True))
        for images, labels, _ in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            step_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{step_lr:.6f}")

        train_acc = 100 * correct / total
        logging.info(f'[Train] [{epoch+1}/{train_cfg.epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')
        
        eval_results = evaluate(run_cfg, model, valid_loader, device, desc=f"[Valid] [{epoch+1}/{train_cfg.epochs}]", class_names=class_names, log_class_metrics=True)
        
        current_f1 = 0.0
        if best_model_criterion == 'F1_Normal' and eval_results['f1_per_class'] is not None:
            current_f1 = eval_results['f1_per_class'][0]
        elif best_model_criterion == 'F1_Defect' and eval_results['f1_per_class'] is not None:
            current_f1 = eval_results['f1_per_class'][1]
        else:
            current_f1 = eval_results['f1_macro']
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), model_path)
            criterion_name = best_model_criterion.replace('_', ' ')
            logging.info(f"[Best Model Saved] ({criterion_name}: {best_f1:.4f}) -> '{model_path}'")
        
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
    if torch.cuda.is_available():
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
        logging.info("CUDA를 사용할 수 없어 GPU 메모리 사용량 및 정확한 추론 시간을 측정하지 않습니다.")
        start_time = time.time()
        _ = model(dummy_input)
        end_time = time.time()
        avg_inference_time_per_sample = (end_time - start_time) * 1000 # ms
        logging.info(f"샘플 당 평균 Forward Pass 시간 (CPU): {avg_inference_time_per_sample:.2f}ms (1회 측정)")
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
        eval_results = evaluate(run_cfg, model, data_loader, device, desc=f"[{mode_name}]", class_names=class_names, log_class_metrics=True)
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
    train_cfg = SimpleNamespace(**config['training'])
    model_cfg = SimpleNamespace(**config['model'])
    baseline_cfg = SimpleNamespace(**config.get('baseline', {})) # baseline 섹션 로드
    run_cfg.dataset = SimpleNamespace(**run_cfg.dataset)
    
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # --- 데이터 준비 ---
    train_loader, valid_loader, test_loader, num_labels, class_names = prepare_data(run_cfg, train_cfg, model_cfg)

    # --- Baseline 모델 생성 ---
    model = create_baseline_model(baseline_model_name, num_labels, pretrained=True).to(device)
    log_model_parameters(model)
    
    # --- 옵티마이저 및 스케줄러 설정 ---
    optimizer, scheduler = None, None
    if run_cfg.mode == 'train':
        if getattr(train_cfg, 'schedulefree', False):
            logging.info("Schedule-Free 옵티마이저 (AdamWScheduleFree)를 사용합니다.")
            optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=train_cfg.lr)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr)
            use_cosine_lr = getattr(train_cfg, 'CosineAnnealingLR', False)
            use_cosine_warm_restarts = getattr(train_cfg, 'CosineAnnealingWarmRestarts', False)

            if use_cosine_lr:
                logging.info(f"표준 옵티마이저 (AdamW)와 CosineAnnealingLR 스케줄러를 사용합니다. (T_max={train_cfg.epochs})")
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
            elif use_cosine_warm_restarts:
                T_0 = getattr(train_cfg, 'T_0', 10)
                T_mult = getattr(train_cfg, 'T_mult', 1)
                logging.info(f"표준 옵티마이저 (AdamW)와 CosineAnnealingWarmRestarts 스케줄러를 사용합니다. (T_0={T_0}, T_mult={T_mult})")
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
            else:
                logging.info("표준 옵티마이저 (AdamW)를 사용합니다. (스케줄러 없음)")
        logging.info("="*50)

    # --- 모드에 따라 실행 ---
    if run_cfg.mode == 'train':
        train(run_cfg, train_cfg, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names)
        logging.info("="*50)
        logging.info("훈련 완료. 최종 모델 성능을 테스트 세트로 평가합니다.")
        final_acc = inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Test", class_names=class_names)

        log_filename = f"log_{timestamp}.log"
        log_file_path = os.path.join(run_dir_path, log_filename)
        if final_acc is not None:
            plot_and_save_val_accuracy_graph(log_file_path, run_dir_path, final_acc, timestamp)
            plot_and_save_train_val_accuracy_graph(log_file_path, run_dir_path, final_acc, timestamp)

    elif run_cfg.mode == 'inference':
        inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=class_names)

if __name__ == '__main__':
    main()