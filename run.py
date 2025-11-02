import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score
from types import SimpleNamespace
import numpy as np
import pandas as pd
from PIL import Image

import argparse
import yaml
import logging
from datetime import datetime
import time
from models import Model as CatsDecoder, PatchConvEncoder, Classifier, HybridModel
import schedulefree

try:
    from thop import profile
except ImportError:
    profile = None

from plot import plot_and_save_train_val_accuracy_graph, plot_and_save_val_accuracy_graph, plot_and_save_confusion_matrix, plot_and_save_attention_maps

# =============================================================================
# 1. 로깅 설정
# =============================================================================
def setup_logging(run_cfg, data_dir_name):
    """로그 파일을 log 폴더에 생성하고, 콘솔에도 함께 출력하도록 설정합니다."""
    show_log = getattr(run_cfg, 'show_log', True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not show_log:
        # 로깅을 완전히 비활성화합니다.
        logging.disable(logging.CRITICAL)
        # 임시 디렉토리 경로를 반환하지만, 실제 생성은 하지 않습니다.
        # 훈련 모드에서 모델 저장을 위해 현재 디렉토리('.')를 사용합니다.
        return '.', timestamp

    # 각 실행을 위한 고유한 디렉토리 생성
    run_dir_name = f"run_{timestamp}"
    run_dir_path = os.path.join("log", data_dir_name, run_dir_name)
    os.makedirs(run_dir_path, exist_ok=True)
    
    # 로그 파일 경로 설정
    log_filename = os.path.join(run_dir_path, f"log_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True # 핸들러를 다시 설정하기 위해 필요
    )
    logging.info(f"로그 파일이 '{log_filename}'에 저장됩니다.")
    return run_dir_path, timestamp

# =============================================================================
# 3. 훈련 및 평가 함수
# =============================================================================
def log_model_parameters(model):
    """모델의 구간별 및 총 파라미터 수를 계산하고 로깅합니다."""
    
    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    # Encoder 내부를 세분화하여 파라미터 계산
    # 1. Encoder (PatchConvEncoder) 내부를 세분화하여 파라미터 계산
    cnn_feature_extractor = model.encoder.shared_conv[0]

    # CNN 백본의 파라미터를 계산합니다.
    conv_front_params = count_parameters(cnn_feature_extractor.conv_front)
    conv_1x1_params = count_parameters(cnn_feature_extractor.conv_1x1)
    encoder_norm_params = count_parameters(model.encoder.norm)
    encoder_total_params = conv_front_params + conv_1x1_params + encoder_norm_params


    # CatsDecoder (models.py의 Model 클래스)의 구성 요소
    # - Embedding4Decoder (W_feat2emb, learnable_queries, PE)
    # - Embedding4Decoder 내부의 Decoder (트랜스포머 레이어들)
    # - Projection4Classifier

    # Embedding4Decoder의 파라미터를 세분화하여 계산
    embedding_module = model.decoder.embedding4decoder
    
    # PE와 learnable_queries는 nn.Parameter이므로 .numel()로 직접 개수 계산
    pe_params = 0 # Positional Encoding
    if hasattr(embedding_module, 'PE') and embedding_module.PE is not None and embedding_module.PE.requires_grad:
        pe_params = embedding_module.PE.numel()
    
    query_params = 0 # Learnable Query
    if hasattr(embedding_module, 'learnable_queries') and embedding_module.learnable_queries.requires_grad:
        query_params = embedding_module.learnable_queries.numel()
    
    w_feat2emb_params = count_parameters(embedding_module.W_feat2emb)

    # Embedding4Decoder의 자체 파라미터 총합 (내부 Decoder 제외)
    embedding4decoder_total_params = w_feat2emb_params + query_params + pe_params

    cats_decoder_layers_params = count_parameters(model.decoder.embedding4decoder.decoder)
    cats_decoder_projection4classifier_params = count_parameters(model.decoder.projection4classifier)
    cats_decoder_total_params = embedding4decoder_total_params + cats_decoder_layers_params + cats_decoder_projection4classifier_params

    # 3. Classifier (Linear Head) 내부를 세분화하여 파라미터 계산
    classifier_projection_params = count_parameters(model.classifier.projection)
    classifier_total_params = classifier_projection_params

    total_params = encoder_total_params + cats_decoder_total_params + classifier_total_params

    logging.info("="*50)
    logging.info("모델 파라미터 수:")
    logging.info(f"  - Encoder (PatchConvEncoder):   {encoder_total_params:,} 개")
    logging.info(f"    - conv_front (CNN Backbone):  {conv_front_params:,} 개")
    logging.info(f"    - 1x1_conv (Channel Proj):    {conv_1x1_params:,} 개")
    logging.info(f"    - norm (LayerNorm):           {encoder_norm_params:,} 개")
    logging.info(f"  - Decoder (CatsDecoder):        {cats_decoder_total_params:,} 개")
    logging.info(f"    - Embedding Layer (W_feat2emb): {w_feat2emb_params:,} 개")
    logging.info(f"    - Learnable Query:              {query_params:,} 개")
    logging.info(f"    - Positional Encoding:             {pe_params:,} 개")
    logging.info(f"    - Decoder Layers (Cross-Attention): {cats_decoder_layers_params:,} 개")
    logging.info(f"    - Projection4Classifier:      {cats_decoder_projection4classifier_params:,} 개")
    logging.info(f"  - Classifier (Projection MLP):  {classifier_total_params:,} 개")
    logging.info(f"  - 총 파라미터:                  {total_params:,} 개")
    logging.info("="*50)

def _get_model_weights_norm(model):
    """모델의 모든 학습 가능한 파라미터에 대한 L2 Norm을 계산합니다."""
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.detach().norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def evaluate(run_cfg, model, optimizer, data_loader, device, desc="Evaluating", class_names=None, log_class_metrics=False):
    """모델을 평가하고 정확도, 정밀도, 재현율, F1 점수를 로깅합니다."""
    model.eval()

    use_schedulefree = optimizer and hasattr(optimizer, 'eval')
    if use_schedulefree:
        norm_before = _get_model_weights_norm(model)

    # schedulefree 옵티마이저를 위해 optimizer도 eval 모드로 설정
    if optimizer and hasattr(optimizer, 'eval'):
        optimizer.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    total_forward_time = 0.0
    
    show_log = getattr(run_cfg, 'show_log', True)
    progress_bar = tqdm(data_loader, desc=desc, leave=False, disable=not show_log)
    with torch.no_grad():
        for images, labels, _ in progress_bar: # 파일명은 사용하지 않으므로 _로 받음
            images, labels = images.to(device), labels.to(device)

            # --- 순수 forward pass 시간 측정 ---
            if device.type == 'cuda':
                # GPU 시간 측정을 위한 이벤트 생성
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record() # 시작 시간 기록
                outputs = model(images) # [B, num_labels]
                end_event.record() # 종료 시간 기록

                torch.cuda.synchronize() # GPU 연산이 끝날 때까지 대기
                
                # 밀리초(ms) 단위의 시간을 초 단위로 변환하여 누적
                total_forward_time += start_event.elapsed_time(end_event) / 1000.0
            else: # CPU의 경우 time.time() 사용
                start_time = time.time()
                outputs = model(images)
                end_time = time.time()
                total_forward_time += (end_time - start_time)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    if use_schedulefree:
        norm_after = _get_model_weights_norm(model)
        logging.info(f'[Schedule-Free] 가중치 업데이트의 안정화. 가중치의 L2 Norm: {norm_before:.4f} -> {norm_after:.4f}')

    if total == 0:
        logging.warning("테스트 데이터가 없습니다. 평가를 건너뜁니다.")
        return {'accuracy': 0.0, 'f1': 0.0, 'labels': [], 'preds': [], 'forward_time': 0.0}

    # desc 내용에 따라 Accuracy 라벨을 동적으로 변경
    if desc.startswith("[Valid]"):
        acc_label = "Val Acc"
        log_message = f'{desc} | {acc_label}: {accuracy:.2f}%'
    else: # [Test] 또는 [Inference]의 경우
        acc_label = "Test Acc"
        log_message = f'{desc} {acc_label}: {accuracy:.2f}%'
    logging.info(log_message)

    # 클래스별 상세 지표 로깅
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
        'f1_macro': f1, # 평균 F1 점수
        'f1_per_class': f1_per_class if log_class_metrics and class_names else None, # 클래스별 F1 점수
        'labels': all_labels,
        'preds': all_preds,
        'forward_time': total_forward_time
    }

def train(run_cfg, train_cfg, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names):
    """모델 훈련 및 검증을 수행하고 최고 성능 모델을 저장합니다."""
    logging.info("train 모드를 시작합니다.")
    
    # 모델 저장 경로를 실행별 디렉토리로 설정
    model_path = os.path.join(run_dir_path, run_cfg.model_path)

    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0
    best_model_criterion = getattr(train_cfg, 'best_model_criterion', 'F1_average')

    for epoch in range(train_cfg.epochs):
        # 에포크 시작 시 구분을 위한 라인 추가
        logging.info("-" * 50)

        model.train()
        # schedulefree 옵티마이저를 위해 optimizer도 train 모드로 설정
        if optimizer and hasattr(optimizer, 'train'):
            optimizer.train()

        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.epochs} [Training]", leave=False, disable=not getattr(run_cfg, 'show_log', True))
        for images, labels, _ in progress_bar: # 파일명은 사용하지 않으므로 _로 받음
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
            
            # tqdm 프로그레스 바에 현재 loss 표시
            step_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{step_lr:.6f}")

        train_acc = 100 * correct / total
        logging.info(f'[Train] [{epoch+1}/{train_cfg.epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')
        
        # --- 평가 단계 ---
        # 클래스별 F1 점수를 계산하고 로깅하도록 옵션 전달
        eval_results = evaluate(run_cfg, model, optimizer, valid_loader, device, desc=f"[Valid] [{epoch+1}/{train_cfg.epochs}]", class_names=class_names, log_class_metrics=True)
        
        # --- 최고 성능 모델 저장 기준 선택 ---
        current_f1 = 0.0
        if best_model_criterion == 'F1_Normal' and eval_results['f1_per_class'] is not None:
            current_f1 = eval_results['f1_per_class'][0] # 'Normal' 클래스는 인덱스 0
        elif best_model_criterion == 'F1_Defect' and eval_results['f1_per_class'] is not None:
            current_f1 = eval_results['f1_per_class'][1] # 'Defect' 클래스는 인덱스 1
        else: # 'F1_average' 또는 그 외
            current_f1 = eval_results['f1_macro']
        
        # 최고 성능 모델 저장
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), model_path)
            # 어떤 기준으로 저장되었는지 명확히 로그에 남깁니다.
            criterion_name = best_model_criterion.replace('_', ' ')
            logging.info(f"[Best Model Saved] ({criterion_name}: {best_f1:.4f}) -> '{model_path}'")
        
        # 스케줄러가 설정된 경우에만 step()을 호출
        if scheduler:
            scheduler.step()

def inference(run_cfg, model_cfg, cats_cfg, model, optimizer, data_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=None):
    """저장된 모델을 불러와 추론 시 GPU 메모리 사용량을 측정하고, 테스트셋 성능을 평가합니다."""
    logging.info(f"{mode_name} 모드를 시작합니다.")
    
    # 훈련 시 사용된 모델 경로를 불러옴
    model_path = os.path.join(run_dir_path, run_cfg.model_path)
    if not os.path.exists(model_path) and mode_name != "Final Evaluation":
        logging.error(f"모델 파일('{model_path}')을 찾을 수 없습니다. 'train' 모드로 먼저 훈련을 실행했는지, 또는 'config.yaml'의 'run_dir_for_inference' 설정이 올바른지 확인하세요.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logging.info(f"'{model_path}' 가중치 로드 완료.")
    except Exception as e:
        logging.error(f"모델 가중치 로딩 중 오류 발생: {e}")
        return

    model.eval()

    use_schedulefree = optimizer and hasattr(optimizer, 'eval')
    if use_schedulefree:
        norm_before = _get_model_weights_norm(model)

    # schedulefree 옵티마이저를 위해 optimizer도 eval 모드로 설정
    # 최종 가중치를 모델 파라미터에 통합(consolidate)합니다.
    if optimizer and hasattr(optimizer, 'eval'):
        optimizer.eval()

    # --- FLOPS 측정 ---
    # 모델의 입력 크기를 확인하기 위해 샘플 이미지를 하나 가져옵니다.
    # test_loader.dataset은 Subset일 수 있으므로 .dataset으로 원본 데이터셋에 접근합니다.
    try:
        sample_image, _, _ = data_loader.dataset.dataset[0] if isinstance(data_loader.dataset, Subset) else data_loader.dataset[0]
        dummy_input = sample_image.unsqueeze(0).to(device)

        if profile:
            # thop.profile은 MACs를 반환합니다. FLOPS는 보통 MACs * 2 입니다.
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
            # GFLOPS (Giga Floating Point Operations) 단위로 변환
            gflops = (macs * 2) / 1e9
            logging.info(f"FLOPS: {gflops:.2f} GFLOPS")
        else:
            logging.info("FLOPS: N/A (thop 라이브러리가 설치되지 않아 측정을 건너뜁니다.)")
            logging.info("  - FLOPS를 측정하려면 'pip install thop'을 실행하세요.")
    except Exception as e:
        logging.error(f"FLOPS 측정 중 오류 발생: {e}")

    
    # 1. GPU 메모리 사용량 측정
    dummy_input = torch.randn(1, model_cfg.in_channels, model_cfg.img_size, model_cfg.img_size).to(device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        logging.info(f"샘플 당 Forward Pass 시 최대 GPU 메모리 사용량: {peak_memory_mb:.2f} MB")
    else:
        logging.info("CUDA를 사용할 수 없어 GPU 메모리 사용량을 측정하지 않습니다.")

    # 2. 테스트셋 성능 평가
    logging.info("테스트 데이터셋에 대한 추론을 시작합니다...")
    
    only_inference_mode = getattr(run_cfg, 'only_inference', False)

    if only_inference_mode:
        # 순수 추론 모드: 예측 결과만 생성하고 CSV로 저장
        all_filenames = []
        all_predictions = []
        all_confidences = []
        show_log = getattr(run_cfg, 'show_log', True)
        progress_bar = tqdm(data_loader, desc=f"[{mode_name}]", leave=False, disable=not show_log)
        with torch.no_grad():
            for images, _, filenames in progress_bar:
                images = images.to(device)
                outputs = model(images)
                
                # Softmax를 적용하여 확률 계산
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 가장 높은 확률(confidence)과 해당 인덱스(예측 클래스)를 가져옴
                confidences, predicted_indices = torch.max(probabilities, 1)
                
                all_filenames.extend(filenames)
                all_predictions.extend([class_names[p] for p in predicted_indices.cpu().numpy()])
                all_confidences.extend(confidences.cpu().numpy())
        
        # 결과를 DataFrame으로 만들어 CSV 파일로 저장
        results_df = pd.DataFrame({
            'filename': all_filenames,
            'prediction': all_predictions,
            'confidence': all_confidences
        })
        results_df['confidence'] = results_df['confidence'].map('{:.4f}'.format) # 소수점 4자리까지 표시
        result_csv_path = os.path.join(run_dir_path, f'inference_results_{timestamp}.csv')
        results_df.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"추론 결과가 '{result_csv_path}'에 저장되었습니다.")
        final_acc = None # 정확도 없음

    else:
        # 평가 모드: 기존 evaluate 함수 호출
        eval_results = evaluate(run_cfg, model, optimizer, data_loader, device, desc=f"[{mode_name}]", class_names=class_names, log_class_metrics=True)
        
        # evaluate 함수에서 반환된 순수 forward pass 시간 사용
        pure_inference_time = eval_results.get('forward_time', 0.0)
        num_test_samples = len(data_loader.dataset)
        avg_inference_time_per_sample = (pure_inference_time / num_test_samples) * 1000 if num_test_samples > 0 else 0
        
        logging.info(f"총 Forward Pass 시간: {pure_inference_time:.2f}s (테스트 샘플 {num_test_samples}개)")
        logging.info(f"샘플 당 평균 Forward Pass 시간: {avg_inference_time_per_sample:.2f}ms")
        final_acc = eval_results['accuracy']

        # 3. 혼동 행렬 생성 및 저장 (최종 평가 시에만)
        if eval_results['labels'] and eval_results['preds']:
            plot_and_save_confusion_matrix(eval_results['labels'], eval_results['preds'], class_names, run_dir_path, timestamp)

    # 4. 어텐션 맵 시각화 (설정이 True인 경우)
    if cats_cfg.save_attention:
        try:
            # 1. 어텐션 맵을 저장할 전용 폴더 생성
            attn_save_dir = os.path.join(run_dir_path, f'attention_map_{timestamp}')
            os.makedirs(attn_save_dir, exist_ok=True)

            num_to_save = getattr(cats_cfg, 'num_plot_attention', 10)
            logging.info(f"어텐션 맵 시각화를 시작합니다 (최대 {num_to_save}개 샘플, 저장 위치: '{attn_save_dir}').")

            # 시각화를 위해 테스트 로더에서 첫 번째 배치를 가져옴
            sample_images, sample_labels, sample_filenames = next(iter(data_loader))
            sample_images = sample_images.to(device)
            batch_size = sample_images.size(0)

            # 모델을 실행하여 어텐션 맵이 저장되도록 함
            with torch.no_grad():
                outputs = model(sample_images)

            _, predicted_indices = torch.max(outputs.data, 1)
            attention_maps = model.decoder.embedding4decoder.decoder.layers[-1].attn

            # 배치에서 최대 num_plot_attention개의 샘플에 대해 어텐션 맵 저장
            num_samples_to_plot = min(num_to_save, batch_size)
            for i in range(num_samples_to_plot):
                predicted_class = class_names[predicted_indices[i].item()]
                original_filename = sample_filenames[i]
                
                # only_inference 모드에서는 실제 클래스를 모름
                if only_inference_mode:
                    actual_class = "Unknown"
                else:
                    actual_class = class_names[sample_labels[i].item()]

                plot_and_save_attention_maps(
                    attention_maps, sample_images, attn_save_dir, model_cfg.img_size, cats_cfg,
                    sample_idx=i, original_filename=original_filename, actual_class=actual_class, predicted_class=predicted_class
                )
        except Exception as e:
            logging.error(f"어텐션 맵 시각화 중 오류 발생: {e}")
    return final_acc

# =============================================================================
# 4. 데이터 준비 함수
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
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        # 'Defect' 열의 값을 명시적으로 레이블로 사용합니다.
        label = int(self.img_labels.loc[idx, 'Defect'])
        return image, label, img_name

class InferenceImageDataset(Dataset):
    """정답 레이블 없이, 지정된 폴더의 모든 이미지를 로드하는 추론 전용 데이터셋입니다."""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # 지원하는 이미지 확장자
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        self.img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in self.image_extensions]
        if not self.img_files:
            logging.warning(f"'{img_dir}'에서 이미지를 찾을 수 없습니다.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image, -1, img_name # 레이블은 -1과 같은 placeholder 값으로 반환

def prepare_data(run_cfg, train_cfg, model_cfg, data_dir_name):
    """데이터셋을 로드하고 전처리하여 DataLoader를 생성합니다."""
    img_size = model_cfg.img_size

    # --- 데이터 샘플링 로직 함수 ---
    def get_subset(dataset, name, sampling_ratios, random_seed):
        """데이터셋에서 지정된 비율만큼 단순 랜덤 샘플링을 수행합니다."""
        # sampling_ratios가 딕셔너리인지 확인하고, 해당 데이터셋의 비율을 가져옵니다.
        ratio = 1.0
        if isinstance(sampling_ratios, dict):
            ratio = sampling_ratios.get(name, 1.0)
        elif isinstance(sampling_ratios, (float, int)): # 이전 버전 호환성
            ratio = sampling_ratios

        if ratio < 1.0:
            logging.info(f"'{name}' 데이터셋을 {ratio * 100:.1f}% 비율로 샘플링합니다 (random_seed={random_seed}).")
            num_total = len(dataset)
            num_to_sample = int(num_total * ratio)
            # num_to_sample이 0이 되지 않도록 최소 1개는 샘플링
            num_to_sample = max(1, num_to_sample)
            rng = np.random.default_rng(random_seed) # 재현성을 위한 랜덤 생성기
            indices = rng.choice(num_total, size=num_to_sample, replace=False)
            return Subset(dataset, indices)
        return dataset

    if model_cfg.in_channels == 1:
        # 흑백 이미지용 변환 (커스텀 트랜스폼)
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize
        ])
        valid_test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize
        ])
    else: # model_cfg.in_channels == 3:
        # 컬러 이미지(in_channels=3)용 변환 (Sewer-ML 공식 레포의 트랜스폼)
        normalize = transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
        valid_test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])

    try:
        logging.info("데이터 로드를 시작합니다.")
        
        # only_inference 모드인 경우, 레이블 없이 이미지만 로드
        if run_cfg.mode == 'inference' and getattr(run_cfg, 'only_inference', False):
            logging.info("'only_inference' 모드가 활성화되었습니다. 레이블 없이 추론을 진행합니다.")
            full_test_dataset = InferenceImageDataset(img_dir=run_cfg.test_img_dir, transform=valid_test_transform)
            
            # --- only_inference 모드에서도 샘플링 적용 ---
            sampling_ratios = getattr(run_cfg, 'random_sampling_ratio', None)
            test_dataset = get_subset(full_test_dataset, 'test', sampling_ratios, run_cfg.random_seed)
            
            test_loader = DataLoader(test_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=run_cfg.num_workers, pin_memory=True)
            
            logging.info(f"총 {len(full_test_dataset)}개의 이미지 파일 중 {len(test_dataset)}개를 샘플링하여 추론합니다.")
            
            # only_inference 모드에서는 train/valid 로더가 필요 없으므로 None 반환
            return None, None, test_loader, 2, ['Normal', 'Defect']


        # 커스텀 데이터셋 사용
        full_train_dataset = CustomImageDataset(csv_file=run_cfg.train_csv, img_dir=run_cfg.train_img_dir, transform=train_transform)
        full_valid_dataset = CustomImageDataset(csv_file=run_cfg.valid_csv, img_dir=run_cfg.valid_img_dir, transform=valid_test_transform)
        full_test_dataset = CustomImageDataset(csv_file=run_cfg.test_csv, img_dir=run_cfg.test_img_dir, transform=valid_test_transform)

        # ----레이블 유효성 검사 ---
        def validate_labels(dataset, name):
            if 'Defect' not in dataset.img_labels.columns:
                raise ValueError(f"{name} 데이터셋의 CSV 파일에 'Defect' 열이 없습니다.")
            labels = dataset.img_labels['Defect']
            unique_labels = labels.unique()
            if not all(label in [0, 1] for label in unique_labels):
                invalid_labels = [label for label in unique_labels if label not in [0, 1]]
                raise ValueError(f"{name} 데이터셋의 CSV 파일에 유효하지 않은 레이블이 포함되어 있습니다: {invalid_labels}. 레이블은 0 또는 1이어야 합니다.")
        
        validate_labels(full_train_dataset, "Train")
        validate_labels(full_valid_dataset, "Validation")
        validate_labels(full_test_dataset, "Test")
        # ------

        num_labels = 2 # 0과 1의 이진 분류
        class_names = ['Normal', 'Defect']

        # --- 훈련/검증/테스트 데이터 샘플링 ---
        sampling_ratios = getattr(run_cfg, 'random_sampling_ratio', None)
        train_dataset = get_subset(full_train_dataset, 'train', sampling_ratios, run_cfg.random_seed)
        valid_dataset = get_subset(full_valid_dataset, 'valid', sampling_ratios, run_cfg.random_seed)
        test_dataset = get_subset(full_test_dataset, 'test', sampling_ratios, run_cfg.random_seed)

        # DataLoader 생성
        train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=run_cfg.num_workers, pin_memory=True, persistent_workers=True if run_cfg.num_workers > 0 else False)
        valid_loader = DataLoader(valid_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=run_cfg.num_workers, pin_memory=True, persistent_workers=True if run_cfg.num_workers > 0 else False)
        test_loader = DataLoader(test_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=run_cfg.num_workers, pin_memory=True, persistent_workers=True if run_cfg.num_workers > 0 else False)
        
        logging.info(f"훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(valid_dataset)}개, 테스트 데이터: {len(test_dataset)}개")
        
        return train_loader, valid_loader, test_loader, num_labels, class_names
        
    except FileNotFoundError:
        logging.error(f"데이터 폴더 또는 CSV 파일을 찾을 수 없습니다. 'run.yaml'의 경로 설정을 확인해주세요.")
        exit()


def main():
    """메인 실행 함수"""
    # --- YAML 설정 파일 로드 ---
    parser = argparse.ArgumentParser(description="YAML 설정을 이용한 CATS 기반 이미지 분류기")
    parser.add_argument('--config', type=str, default='config.yaml', help='설정 파일 경로')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # SimpleNamespace를 사용하여 딕셔너리처럼 접근 가능하게 변환
    run_cfg = SimpleNamespace(**config['run'])
    train_cfg = SimpleNamespace(**config['training'])
    # model_cfg와 cats_cfg는 model_cfg 내부에 cats가 포함된 구조이므로 아래와 같이 파싱
    model_cfg = SimpleNamespace(**config['model'])
    cats_cfg = SimpleNamespace(**model_cfg.cats)

    data_dir_name = os.path.basename(os.path.normpath(os.path.dirname(run_cfg.train_img_dir)))

    # --- 실행 디렉토리 설정 ---
    if run_cfg.mode == 'train':
        # 훈련 모드: 새로운 실행 디렉토리 생성
        run_dir_path, timestamp = setup_logging(run_cfg, data_dir_name) # 여기서 run_dir_path와 timestamp가 반환됨
    elif run_cfg.mode == 'inference':
        # 추론 모드: 지정된 실행 디렉토리 사용
        run_dir_path = getattr(run_cfg, 'run_dir_for_inference', None)
        if getattr(run_cfg, 'show_log', True) and (not run_dir_path or not os.path.isdir(run_dir_path)):
            logging.error("추론 모드에서는 'config.yaml'에 'run_dir_for_inference'를 올바르게 설정해야 합니다.")
            exit()
        # 로깅 설정은 하지만, run_dir_path는 yaml에서 읽은 값을 사용
        _, timestamp = setup_logging(run_cfg, data_dir_name)
    
    # --- 설정 파일 내용 로깅 ---
    config_str = yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logging.info("="*50)
    logging.info("config.yaml:")
    logging.info("\n" + config_str)
    logging.info("="*50)
    
    # --- 공통 파라미터 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        logging.info(f"CUDA 사용 가능. GPU 사용을 시작합니다. (Device: {torch.cuda.get_device_name(0)})")
    else:
        logging.info("CUDA 사용 불가능. CPU 사용을 시작합니다.")

    # --- 데이터 준비 ---
    train_loader, valid_loader, test_loader, num_labels, class_names = prepare_data(run_cfg, train_cfg, model_cfg, "Sewer-ML")

    # --- 모델 구성 ---
    num_encoder_patches = (model_cfg.img_size // model_cfg.patch_size) ** 2 # 16
    
    cats_params = {
        'num_encoder_patches': num_encoder_patches,
        'num_labels': num_labels, 
        'num_decoder_layers': cats_cfg.num_decoder_layers,
        'num_decoder_patches': cats_cfg.num_decoder_patches, # YAML에서 읽은 값 전달
        'featured_patch_dim': cats_cfg.featured_patch_dim,
        'attn_pooling': getattr(cats_cfg, 'attn_pooling', False), # 어텐션 풀링 사용 여부
        'emb_dim': cats_cfg.emb_dim, 
        'num_heads': cats_cfg.num_heads, 
        'decoder_ff_ratio': cats_cfg.decoder_ff_ratio,
        'dropout': cats_cfg.dropout, # dropout
        'positional_encoding': cats_cfg.positional_encoding, # positional_encoding
        'res_attention': cats_cfg.res_attention, # res_attention
        'save_attention': cats_cfg.save_attention, # save_attention
        'qam_prob_start': cats_cfg.qam_prob_start, # qam_prob_start
        'qam_prob_end': cats_cfg.qam_prob_end, # qam_prob_end
    }
    cats_args = SimpleNamespace(**cats_params)

    encoder = PatchConvEncoder(in_channels=model_cfg.in_channels, img_size=model_cfg.img_size, patch_size=model_cfg.patch_size, 
                               featured_patch_dim=cats_cfg.featured_patch_dim, cnn_feature_extractor_name=model_cfg.cnn_feature_extractor['name'])
    decoder = CatsDecoder(args=cats_args) # models.py의 Model 클래스
    
    classifier = Classifier(num_decoder_patches=cats_cfg.num_decoder_patches, 
                            featured_patch_dim=cats_cfg.featured_patch_dim, num_labels=num_labels, dropout=cats_cfg.dropout)
    model = HybridModel(encoder, decoder, classifier).to(device)

    # 모델 생성 후 파라미터 수 로깅
    log_model_parameters(model)
    
    # --- 모드에 따라 실행 ---
    if run_cfg.mode == 'train':
        # --- 옵티마이저 및 스케줄러 설정 (훈련 모드에서만) ---
        if getattr(train_cfg, 'schedulefree', False):
            logging.info("Schedule-Free 옵티마이저 (AdamWScheduleFree)를 사용합니다.")
            optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=train_cfg.lr)
            scheduler = None # schedulefree는 스케줄러가 필요 없음
        else:
            # 표준 AdamW 옵티마이저 사용
            optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr)
            scheduler = None # 기본값은 스케줄러 없음
            
            # YAML 설정에 따라 스케줄러를 선택합니다.
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
            
            if scheduler is None:
                logging.info("표준 옵티마이저 (AdamW)를 사용합니다. (스케줄러 없음)")

        logging.info("="*50)

        # 훈련 시에는 train_loader와 valid_loader 사용
        train(run_cfg, train_cfg, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names)

        logging.info("="*50)
        logging.info("훈련 완료. 최종 모델 성능을 테스트 세트로 평가합니다.")
        final_acc = inference(run_cfg, model_cfg, cats_cfg, model, optimizer, test_loader, device, run_dir_path, timestamp, mode_name="Test", class_names=class_names)

        # --- 그래프 생성 ---
        # 로그 파일 이름은 setup_logging에서 생성된 패턴을 기반으로 함
        log_filename = f"log_{timestamp}.log"
        log_file_path = os.path.join(run_dir_path, log_filename)
        if final_acc is not None:
            plot_and_save_val_accuracy_graph(log_file_path, run_dir_path, final_acc, timestamp)
            plot_and_save_train_val_accuracy_graph(log_file_path, run_dir_path, final_acc, timestamp)

    elif run_cfg.mode == 'inference':
        # 추론 모드에서는 test_loader를 사용해 성능 평가
        optimizer, scheduler = None, None # 추론 시에는 옵티마이저/스케줄러가 필요 없음
        inference(run_cfg, model_cfg, cats_cfg, model, optimizer, test_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=class_names)

# =============================================================================
# 5. 메인 실행 블록
# =============================================================================
if __name__ == '__main__':
    main()