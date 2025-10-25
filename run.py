import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
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

# CATS 모델 아키텍처 임포트
from CATS import Model as CatsDecoder

# Schedule-Free 옵티마이저 임포트
import schedulefree

# 그래프 및 혼동 행렬 플로팅 함수 임포트
from plot import plot_and_save_train_val_accuracy_graph, plot_and_save_val_accuracy_graph, plot_and_save_confusion_matrix, plot_and_save_attention_maps

# =============================================================================
# 1. 로깅 설정
# =============================================================================
def setup_logging(data_dir_name):
    """로그 파일을 log 폴더에 생성하고, 콘솔에도 함께 출력하도록 설정합니다."""
    # 각 실행을 위한 고유한 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        ]
    )
    logging.info(f"로그 파일이 '{log_filename}'에 저장됩니다.")
    return run_dir_path

def cutmix(data, targets, alpha=1.0):
    """CutMix를 적용하는 함수"""
    indices = torch.randperm(data.size(0))

# =============================================================================
# 2. 이미지 인코더 모델 정의
# =============================================================================
class CnnFeatureExtractor(nn.Module):
    """
    다양한 CNN 아키텍처의 앞부분을 특징 추출기로 사용하는 범용 클래스입니다.
    run.yaml의 `cnn_feature_extractor.name` 설정에 따라 모델 구조가 결정됩니다.
    """
    def __init__(self, cnn_feature_extractor_name='resnet18_layer1', pretrained=True, in_channels=3, featured_patch_dim=None):
        super().__init__()
        self.cnn_feature_extractor_name = cnn_feature_extractor_name

        # CNN 모델 이름에 따라 모델과 잘라낼 레이어, 기본 출력 채널을 설정합니다.
        if cnn_feature_extractor_name == 'resnet18_layer1':
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.conv_front = nn.Sequential(*list(base_model.children())[:5]) # layer1까지
            base_out_channels = 64
        elif cnn_feature_extractor_name == 'resnet18_layer2':
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.conv_front = nn.Sequential(*list(base_model.children())[:6]) # layer2까지
            base_out_channels = 128
            
            
        elif cnn_feature_extractor_name == 'mobilenet_v3_small_feat1':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.conv_front = base_model.features[:2] # features의 2번째 블록까지
            base_out_channels = 16
        elif cnn_feature_extractor_name == 'mobilenet_v3_small_feat3':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.conv_front = base_model.features[:4] # features의 4번째 블록까지
            base_out_channels = 24
        elif cnn_feature_extractor_name == 'mobilenet_v3_small_feat4':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.conv_front = base_model.features[:5] # features의 5번째 블록까지
            base_out_channels = 40
        # mobilenet_v3_small_feat5부터는 파라미터가 9만개..
            
            
        elif cnn_feature_extractor_name == 'efficientnet_b0_feat2':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.conv_front = base_model.features[:3] # features의 3번째 블록까지
            base_out_channels = 24
        elif cnn_feature_extractor_name == 'efficientnet_b0_feat3':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.conv_front = base_model.features[:4] # features의 4번째 블록까지
            base_out_channels = 40
        else:
            raise ValueError(f"지원하지 않는 CNN 피처 추출기 이름입니다: {cnn_feature_extractor_name}")

        # 최종 출력 채널 수를 `featured_patch_dim`에 맞추기 위한 1x1 컨볼루션 레이어입니다.
        if featured_patch_dim is not None and featured_patch_dim != base_out_channels:
            self.conv_1x1 = nn.Conv2d(base_out_channels, featured_patch_dim, kernel_size=1)
        else:
            self.conv_1x1 = nn.Identity()

    def _adjust_input_channels(self, base_model, in_channels):
        """모델의 첫 번째 컨볼루션 레이어의 입력 채널을 조정합니다."""
        if in_channels == 1:
            # 첫 번째 conv 레이어 찾기
            if 'resnet' in self.cnn_feature_extractor_name:
                first_conv = base_model.conv1
                out_c, _, k, s, p, _, _, _ = first_conv.out_channels, first_conv.in_channels, first_conv.kernel_size, first_conv.stride, first_conv.padding, first_conv.dilation, first_conv.groups, first_conv.bias
                new_conv = nn.Conv2d(1, out_c, kernel_size=k, stride=s, padding=p, bias=False)
                with torch.no_grad():
                    new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                base_model.conv1 = new_conv
            elif 'mobilenet' in self.cnn_feature_extractor_name or 'efficientnet' in self.cnn_feature_extractor_name:
                first_conv = base_model.features[0][0] # nn.Sequential -> Conv2dNormActivation -> Conv2d
                out_c, _, k, s, p, _, _, _ = first_conv.out_channels, first_conv.in_channels, first_conv.kernel_size, first_conv.stride, first_conv.padding, first_conv.dilation, first_conv.groups, first_conv.bias
                new_conv = nn.Conv2d(1, out_c, kernel_size=k, stride=s, padding=p, bias=False)
                with torch.no_grad():
                    new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                base_model.features[0][0] = new_conv
        elif in_channels != 3:
            raise ValueError("in_channels는 1 또는 3만 지원합니다.")

    def forward(self, x):
        x = self.conv_front(x)
        
        x = self.conv_1x1(x) # 최종 채널 수 조정
        return x

class PatchConvEncoder(nn.Module):
    """이미지를 패치로 나누고, 각 패치에서 특징을 추출하여 1D 시퀀스로 변환하는 인코더입니다."""
    def __init__(self, in_channels, img_size, patch_size, featured_patch_dim, cnn_feature_extractor_name):
        super(PatchConvEncoder, self).__init__()
        self.patch_size = patch_size
        self.featured_patch_dim = featured_patch_dim
        self.num_encoder_patches = (img_size // patch_size) ** 2
        
        self.shared_conv = nn.Sequential(
            CnnFeatureExtractor(cnn_feature_extractor_name=cnn_feature_extractor_name, pretrained=True, in_channels=in_channels, featured_patch_dim=featured_patch_dim),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1) # [B*num_encoder_patches, D, 1, 1] -> [B*num_encoder_patches, D] 형태가 됩니다.
        )
        self.norm = nn.LayerNorm(featured_patch_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # .contiguous()를 추가하여 메모리 연속성을 보장한 후 reshape 수행
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, C, self.patch_size, self.patch_size)
        # patches.shape: [B * num_patches, C, patch_size, patch_size]
        
        conv_outs = self.shared_conv(patches)
        # 각 패치 특징 벡터에 대해 Layer Normalization 적용
        conv_outs = self.norm(conv_outs)
        # CATS 모델에 입력하기 위해 [B, num_patches, dim] 형태로 재구성
        conv_outs = conv_outs.view(B, self.num_encoder_patches, self.featured_patch_dim)
        return conv_outs

class Classifier(nn.Module):
    """CATS 모델의 출력을 받아 최종 클래스 로짓으로 매핑하는 분류기입니다."""
    def __init__(self, num_decoder_patches, featured_patch_dim, num_labels, dropout):
        super().__init__()
        input_dim = num_decoder_patches * featured_patch_dim # 48
        hidden_dim = (input_dim + num_labels) // 2 # 중간 은닉층 차원 (예: (48+2)//2 = 25)

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x):
        # x shape: [B, num_decoder_patches * featured_patch_dim]
        x = self.projection(x) # -> [B, num_labels]
        return x

class HybridModel(torch.nn.Module):
    """인코더와 CATS 분류기를 결합한 최종 하이브리드 모델입니다."""
    def __init__(self, encoder, decoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        
    def forward(self, x):
        # 1. 인코딩: 2D 이미지 -> 패치 시퀀스
        x = self.encoder(x)
        # 2. 크로스-어텐션: 패치 시퀀스 -> 특징 벡터
        x = self.decoder(x)
        # 3. 분류: 특징 벡터 -> 클래스 로짓
        out = self.classifier(x)
        return out

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


    # CatsDecoder (CATS.py의 Model 클래스)의 구성 요소
    # - Embedding4Decoder (W_feat2emb, learnable_queries, PE)
    # - Embedding4Decoder 내부의 Decoder (트랜스포머 레이어들)
    # - Projection4Classifier

    # Embedding4Decoder의 파라미터를 세분화하여 계산
    embedding_module = model.decoder.embedding4decoder
    
    # PE와 learnable_queries는 nn.Parameter이므로 .numel()로 직접 개수 계산
    pe_params = 0
    if hasattr(embedding_module, 'PE') and embedding_module.PE is not None and embedding_module.PE.requires_grad:
        pe_params = embedding_module.PE.numel()
    
    queries_params = 0
    if hasattr(embedding_module, 'learnable_queries') and embedding_module.learnable_queries.requires_grad:
        queries_params = embedding_module.learnable_queries.numel()
    
    w_feat2emb_params = count_parameters(embedding_module.W_feat2emb)

    # Embedding4Decoder의 자체 파라미터 총합 (내부 Decoder 제외)
    embedding4decoder_total_params = w_feat2emb_params + queries_params + pe_params

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
    logging.info(f"    - Embedding Layer (W_feat2emb):    {w_feat2emb_params:,} 개")
    logging.info(f"    - Learnable Queries:               {queries_params:,} 개")
    logging.info(f"    - Positional Encoding (PE):        {pe_params:,} 개")
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

def evaluate(model, optimizer, data_loader, device, desc="Evaluating", class_names=None, log_class_metrics=False):
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
    
    progress_bar = tqdm(data_loader, desc=desc, leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
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
        log_message = f'{desc} | {acc_label}: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}'
    else: # [Test] 또는 [Inference]의 경우
        acc_label = "Test Acc"
        log_message = f'{desc} {acc_label}: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}'
    logging.info(log_message)

    # 클래스별 F1 점수 로깅
    if log_class_metrics and class_names:
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        logging.info("-" * 30)
        for i, class_name in enumerate(class_names):
            logging.info(f"  - F1 Score for '{class_name}': {f1_per_class[i]:.4f}")
        logging.info("-" * 30)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'labels': all_labels,
        'preds': all_preds,
        'forward_time': total_forward_time
    }

def train(run_cfg, train_cfg, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path):
    """모델 훈련 및 검증을 수행하고 최고 성능 모델을 저장합니다."""
    logging.info("훈련 모드를 시작합니다.")
    
    # 모델 저장 경로를 실행별 디렉토리로 설정
    model_path = os.path.join(run_dir_path, run_cfg.model_path)

    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0

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
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.epochs} [Training]", leave=False)
        for images, labels in progress_bar:
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
        eval_results = evaluate(model, optimizer, valid_loader, device, desc=f"[Valid] [{epoch+1}/{train_cfg.epochs}]")
        
        # 최고 성능 모델 저장
        if eval_results['f1'] > best_f1:
            best_f1 = eval_results['f1']
            torch.save(model.state_dict(), model_path)
            logging.info(f"[Best Model Saved] (F1 Score: {best_f1:.4f}) -> '{model_path}'")
        
        # 스케줄러가 설정된 경우에만 step()을 호출
        if scheduler:
            scheduler.step()

def inference(run_cfg, model_cfg, model, optimizer, data_loader, device, run_dir_path, mode_name="Inference", class_names=None):
    """저장된 모델을 불러와 추론 시 GPU 메모리 사용량을 측정하고, 테스트셋 성능을 평가합니다."""
    logging.info(f"{mode_name} 모드를 시작합니다.")
    
    # 훈련 시 사용된 모델 경로를 불러옴
    model_path = os.path.join(run_dir_path, run_cfg.model_path)
    if not os.path.exists(model_path) and mode_name != "Final Evaluation":
        logging.error(f"모델 파일('{model_path}')을 찾을 수 없습니다. 'train' 모드로 먼저 훈련을 실행했는지, 또는 'run.yaml'의 'run_dir_for_inference' 설정이 올바른지 확인하세요.")
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
    eval_results = evaluate(model, optimizer, data_loader, device, desc=f"[{mode_name}]", class_names=class_names, log_class_metrics=True)
    
    # evaluate 함수에서 반환된 순수 forward pass 시간 사용
    pure_inference_time = eval_results.get('forward_time', 0.0)
    num_test_samples = len(data_loader.dataset)
    avg_inference_time_per_sample = (pure_inference_time / num_test_samples) * 1000 if num_test_samples > 0 else 0
    
    logging.info(f"총 Forward Pass 시간: {pure_inference_time:.2f}s (테스트 샘플 {num_test_samples}개)")
    logging.info(f"샘플 당 평균 Forward Pass 시간: {avg_inference_time_per_sample:.2f}ms")
    final_acc = eval_results['accuracy']

    # 3. 혼동 행렬 생성 및 저장 (최종 평가 시에만)
    if eval_results['labels'] and eval_results['preds']:
        cm_save_path = os.path.join(run_dir_path, 'confusion_matrix.png')
        plot_and_save_confusion_matrix(eval_results['labels'], eval_results['preds'], class_names, cm_save_path)

    # 4. 어텐션 맵 시각화 (설정이 True인 경우)
    if cats_cfg.save_attention:
        try:
            # 시각화를 위해 테스트 로더에서 첫 번째 배치를 가져옴
            sample_images, _ = next(iter(data_loader))
            sample_images = sample_images.to(device)

            # 모델을 실행하여 어텐션 맵이 저장되도록 함
            with torch.no_grad():
                _ = model(sample_images)

            # 마지막 디코더 레이어에 저장된 어텐션 맵을 가져옴
            attention_maps = model.decoder.embedding4decoder.decoder.layers[-1].attn
            attn_save_path = os.path.join(run_dir_path, 'attention_map.png')
            plot_and_save_attention_maps(attention_maps, sample_images, attn_save_path, model_cfg.img_size)
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
        return image, label

def prepare_data(run_cfg, train_cfg, model_cfg, data_dir_name):
    """데이터셋을 로드하고 전처리하여 DataLoader를 생성합니다."""
    img_size = model_cfg.img_size

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

        # --- 데이터 샘플링 로직 ---
        random_sampling_ratio = getattr(run_cfg, 'random_sampling_ratio', 1.0)
        if random_sampling_ratio < 1.0:
            logging.info(f"데이터셋을 {random_sampling_ratio * 100:.0f}% 비율로 샘플링합니다 (random_seed={run_cfg.random_seed}).")
            
            def get_subset(dataset):
                """데이터셋에서 지정된 비율만큼 단순 랜덤 샘플링을 수행합니다."""
                num_total = len(dataset)
                num_to_sample = int(num_total * random_sampling_ratio)
                rng = np.random.default_rng(run_cfg.random_seed) # 재현성을 위한 랜덤 생성기
                indices = rng.choice(num_total, size=num_to_sample, replace=False)
                return Subset(dataset, indices)

            train_dataset = get_subset(full_train_dataset)
            valid_dataset = get_subset(full_valid_dataset)
            test_dataset = get_subset(full_test_dataset)
        else:
            logging.info("전체 데이터셋을 사용합니다 (random_sampling_ratio=1.0).")
            train_dataset = full_train_dataset
            valid_dataset = full_valid_dataset
            test_dataset = full_test_dataset

        # DataLoader 생성
        train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=run_cfg.num_workers, pin_memory=True, persistent_workers=True if run_cfg.num_workers > 0 else False)
        valid_loader = DataLoader(valid_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=run_cfg.num_workers, pin_memory=True, persistent_workers=True if run_cfg.num_workers > 0 else False)
        test_loader = DataLoader(test_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=run_cfg.num_workers, pin_memory=True, persistent_workers=True if run_cfg.num_workers > 0 else False)
        
        logging.info(f"훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(valid_dataset)}개, 테스트 데이터: {len(test_dataset)}개")
        
        return train_loader, valid_loader, test_loader, num_labels, class_names
        
    except FileNotFoundError:
        logging.error(f"데이터 폴더 또는 CSV 파일을 찾을 수 없습니다. 'run.yaml'의 경로 설정을 확인해주세요.")
        exit()


# =============================================================================
# 5. 메인 실행 블록
# =============================================================================
if __name__ == '__main__':
    # --- YAML 설정 파일 로드 ---
    parser = argparse.ArgumentParser(description="YAML 설정을 이용한 CATS 기반 이미지 분류기")
    parser.add_argument('--config', type=str, default='run.yaml', help='설정 파일 경로')
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
        run_dir_path = setup_logging(data_dir_name) # 여기서 run_dir_path가 반환됨
    elif run_cfg.mode == 'inference':
        # 추론 모드: 지정된 실행 디렉토리 사용
        run_dir_path = getattr(run_cfg, 'run_dir_for_inference', None)
        if not run_dir_path or not os.path.isdir(run_dir_path):
            logging.error("추론 모드에서는 'run.yaml'에 'run_dir_for_inference'를 올바르게 설정해야 합니다.")
            exit()
        # 로깅 설정은 하지만, run_dir_path는 yaml에서 읽은 값을 사용
        _ = setup_logging(data_dir_name)
    
    # --- 설정 파일 내용 로깅 ---
    config_str = yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logging.info("="*50)
    logging.info("run.yaml:")
    logging.info("\n" + config_str)
    logging.info("="*50)
    
    # --- 공통 파라미터 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        logging.info(f"CUDA 사용 가능. GPU를 사용하여 훈련을 시작합니다. (Device: {torch.cuda.get_device_name(0)})")
    else:
        logging.info("CUDA 사용 불가능. CPU를 사용하여 훈련을 시작합니다.")

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
    decoder = CatsDecoder(args=cats_args) # CATS.py의 Model 클래스
    
    classifier = Classifier(num_decoder_patches=cats_cfg.num_decoder_patches, 
                            featured_patch_dim=cats_cfg.featured_patch_dim, num_labels=num_labels, dropout=cats_cfg.dropout)
    model = HybridModel(encoder, decoder, classifier).to(device)

    # 모델 생성 후 파라미터 수 로깅
    log_model_parameters(model)
    
    # --- 옵티마이저 및 스케줄러 설정 ---
    # run.yaml의 schedulefree 설정에 따라 옵티마이저를 선택합니다.
    use_schedulefree = getattr(train_cfg, 'schedulefree', False)
    use_cosine_scheduler = getattr(train_cfg, 'use_cosine_scheduler', False)

    if use_schedulefree:
        logging.info("Schedule-Free 옵티마이저 (AdamWScheduleFree)를 사용합니다.")
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=train_cfg.lr)
        scheduler = None # schedulefree는 스케줄러가 필요 없음
    else:
        # 표준 AdamW 옵티마이저 사용
        optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr)
        
        if use_cosine_scheduler:
            logging.info(f"표준 옵티마이저 (AdamW)와 CosineAnnealingLR 스케줄러를 사용합니다. (T_max={train_cfg.epochs})")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
        else:
            logging.info("표준 옵티마이저 (AdamW)를 사용합니다. (스케줄러 없음)")
            scheduler = None # 스케줄러를 사용하지 않음

    # --- 모드에 따라 실행 ---
    if run_cfg.mode == 'train':
        # 훈련 시에는 train_loader와 valid_loader 사용
        train(run_cfg, train_cfg, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path)

        logging.info("="*50)
        logging.info("훈련 완료. 최종 모델 성능을 테스트 세트로 평가합니다.")
        final_acc = inference(run_cfg, model_cfg, model, optimizer, test_loader, device, run_dir_path, mode_name="Test", class_names=class_names)

        # --- 그래프 생성 ---
        # 로그 파일 이름은 setup_logging에서 생성된 패턴을 기반으로 함
        log_filename = f"log_{os.path.basename(run_dir_path).replace('run_', '')}.log"
        log_file_path = os.path.join(run_dir_path, log_filename)
        if final_acc is not None:
            plot_and_save_val_accuracy_graph(log_file_path, run_dir_path, final_acc)
            plot_and_save_train_val_accuracy_graph(log_file_path, run_dir_path, final_acc)

    elif run_cfg.mode == 'inference':
        # 추론 모드에서는 test_loader를 사용해 성능 평가
        inference(run_cfg, model_cfg, model, optimizer, test_loader, device, run_dir_path, mode_name="Inference", class_names=class_names)