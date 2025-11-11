import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from sklearn.metrics import precision_score, recall_score, f1_score
from types import SimpleNamespace
import pandas as pd

import argparse
import yaml
import logging
from datetime import datetime
import time 
from models import Model as DecoderBackbone, PatchConvEncoder, Classifier, HybridModel
from dataloader import prepare_data # 데이터 로딩 함수 임포트
import schedulefree

try:
    from thop import profile
except ImportError:
    profile = None

from plot import plot_and_save_train_val_accuracy_graph, plot_and_save_val_accuracy_graph, plot_and_save_confusion_matrix, plot_and_save_attention_maps, plot_and_save_f1_normal_graph, plot_and_save_loss_graph, plot_and_save_lr_graph, plot_and_save_compiled_graph

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

    # 2. Decoder (DecoderBackbone) 내부를 세분화하여 파라미터 계산
    # DecoderBackbone (models.py의 Model 클래스)의 구성 요소
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

    decoder_layers_params = count_parameters(model.decoder.embedding4decoder.decoder)
    decoder_projection4classifier_params = count_parameters(model.decoder.projection4classifier)
    decoder_total_params = embedding4decoder_total_params + decoder_layers_params + decoder_projection4classifier_params

    # 3. Classifier (Projection MLP) 내부를 세분화하여 파라미터 계산
    classifier_projection_params = count_parameters(model.classifier.projection)
    classifier_total_params = classifier_projection_params

    total_params = encoder_total_params + decoder_total_params + classifier_total_params

    logging.info("="*50)
    logging.info("모델 파라미터 수:")
    logging.info(f"  - Encoder (PatchConvEncoder):   {encoder_total_params:,} 개")
    logging.info(f"    - conv_front (CNN Backbone):  {conv_front_params:,} 개")
    logging.info(f"    - 1x1_conv (Channel Proj):    {conv_1x1_params:,} 개")
    logging.info(f"    - norm (LayerNorm):           {encoder_norm_params:,} 개")
    logging.info(f"  - Decoder (Transformer-based):  {decoder_total_params:,} 개")
    logging.info(f"    - Embedding Layer (W_feat2emb): {w_feat2emb_params:,} 개")
    logging.info(f"    - Learnable Queries:            {query_params:,} 개")
    logging.info(f"    - Positional Encoding:          {pe_params:,} 개")
    logging.info(f"    - Decoder Layers (Cross-Attention): {decoder_layers_params:,} 개")
    logging.info(f"    - Projection4Classifier:      {decoder_projection4classifier_params:,} 개")
    logging.info(f"  - Classifier (Projection MLP):  {classifier_total_params:,} 개")
    logging.info(f"  - 총 파라미터:                  {total_params:,} 개")
    logging.info("="*50)

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
        for images, labels, _ in progress_bar: # 파일명은 사용하지 않으므로 _로 받음
            images, labels = images.to(device), labels.to(device)

            outputs = model(images) # [B, num_labels]

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    if total == 0:
        logging.warning("테스트 데이터가 없습니다. 평가를 건너뜁니다.")
        return {'accuracy': 0.0, 'f1': 0.0, 'labels': [], 'preds': []}

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
        # logging.info("-" * 30)
        for i, class_name in enumerate(class_names):
            log_line = (f"[Metrics for '{class_name}'] | "
                        f"Precision: {precision_per_class[i]:.4f} | "
                        f"Recall: {recall_per_class[i]:.4f} | "
                        f"F1: {f1_per_class[i]:.4f}")
            logging.info(log_line)
        # logging.info("-" * 30)

    return {
        'accuracy': accuracy,
        'f1_macro': f1, # 평균 F1 점수
        'f1_per_class': f1_per_class if log_class_metrics and class_names else None, # 클래스별 F1 점수
        'labels': all_labels,
        'preds': all_preds
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
            step_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{step_lr:.6f}")

        train_acc = 100 * correct / total
        logging.info(f'[Train] [{epoch+1}/{train_cfg.epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')
        
        # --- 평가 단계 ---
        # 클래스별 F1 점수를 계산하고 로깅하도록 옵션 전달
        eval_results = evaluate(run_cfg, model, valid_loader, device, desc=f"[Valid] [{epoch+1}/{train_cfg.epochs}]", class_names=class_names, log_class_metrics=True)
        
        # 에포크 종료 시 Learning Rate 로깅
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        logging.info(f"[LR] [{epoch+1}/{train_cfg.epochs}] | Learning Rate: {current_lr:.6f}")

        # --- 최고 성능 모델 저장 기준 선택 ---
        current_f1 = 0.0
        if best_model_criterion == 'F1_Normal' and eval_results['f1_per_class'] is not None:
            try:
                normal_idx = [i for i, name in enumerate(class_names) if name.lower() == 'normal'][0]
                current_f1 = eval_results['f1_per_class'][normal_idx]
            except IndexError:
                logging.warning("best_model_criterion이 'F1_Normal'로 설정되었으나, 'normal' 클래스를 찾을 수 없습니다. 대신 F1_macro를 사용합니다.")
                current_f1 = eval_results['f1_macro']
        elif best_model_criterion == 'F1_Defect' and eval_results['f1_per_class'] is not None:
            try:
                # 'normal'이 아닌 다른 클래스를 'defect'로 간주합니다.
                # 이 로직은 이진 분류(normal vs. one defect class)에 적합합니다.
                defect_idx = [i for i, name in enumerate(class_names) if name.lower() != 'normal'][0]
                current_f1 = eval_results['f1_per_class'][defect_idx]
            except IndexError:
                logging.warning("best_model_criterion이 'F1_Defect'로 설정되었으나, 'defect' 또는 'abnormal' 클래스를 찾을 수 없습니다. 대신 F1_macro를 사용합니다.")
                current_f1 = eval_results['f1_macro']
        elif eval_results['f1_per_class'] is not None: # 'F1_average' 또는 그 외
            current_f1 = eval_results['f1_macro']
        else: # Fallback if f1_per_class is None
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

def inference(run_cfg, model_cfg, model, data_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=None):
    """저장된 모델을 불러와 추론 시 GPU 메모리 사용량을 측정하고, 테스트셋 성능을 평가합니다."""
    logging.info(f"{mode_name} 모드를 시작합니다.")
    
    # 훈련 시 사용된 모델 경로를 불러옴
    model_path = os.path.join(run_dir_path, run_cfg.model_path)
    if not os.path.exists(model_path) and mode_name != "Final Evaluation":
        logging.error(f"모델 파일('{model_path}')을 찾을 수 없습니다. 'train' 모드로 먼저 훈련을 실행했는지, 또는 'config.yaml'의 'only_inference_dir' 설정이 올바른지 확인하세요.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logging.info(f"'{model_path}' 가중치 로드 완료.")
    except Exception as e:
        logging.error(f"모델 가중치 로딩 중 오류 발생: {e}")
        return

    model.eval()

    # --- FLOPS 측정 ---
    # 모델의 입력 크기를 확인하기 위해 샘플 이미지를 하나 가져옵니다.
    # test_loader.dataset은 Subset일 수 있으므로 .dataset으로 원본 데이터셋에 접근합니다.
    gflops_per_sample = 0.0 # 샘플 당 연산량 (FLOPs)
    try:
        sample_image, _, _ = data_loader.dataset.dataset[0] if isinstance(data_loader.dataset, Subset) else data_loader.dataset[0]
        dummy_input = sample_image.unsqueeze(0).to(device)

        if profile:
            # thop.profile은 MACs를 반환합니다. FLOPs는 보통 MACs * 2 입니다.
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
            gmacs = macs / 1e9
            logging.info(f"연산량 (MACs): {gmacs:.2f} GMACs per sample")
            # GFLOPS (Giga Floating Point Operations) 단위로 변환
            gflops_per_sample = (macs * 2) / 1e9
            logging.info(f"연산량 (FLOPs): {gflops_per_sample:.2f} GFLOPs per sample")
        else:
            logging.info("연산량 (FLOPs): N/A (thop 라이브러리가 설치되지 않아 측정을 건너뜁니다.)")
            logging.info("  - FLOPS를 측정하려면 'pip install thop'을 실행하세요.")
    except Exception as e:
        logging.error(f"FLOPS 측정 중 오류 발생: {e}")

    # --- 샘플 당 Forward Pass 시간 및 메모리 사용량 측정 ---
    # FLOPs 측정에 사용된 더미 입력을 재사용합니다.
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
        # 구간별 시간 측정을 위한 이벤트 생성
        start_event = torch.cuda.Event(enable_timing=True)
        encoder_end_event = torch.cuda.Event(enable_timing=True)
        decoder_end_event = torch.cuda.Event(enable_timing=True)
        classifier_end_event = torch.cuda.Event(enable_timing=True)

        num_iterations = 100
        total_times = {'encoder': 0.0, 'decoder': 0.0, 'classifier': 0.0, 'total': 0.0}

        with torch.no_grad():
            for _ in range(num_iterations):
                start_event.record()
                # 1. Encoder 구간
                encoded_features = model.encoder(dummy_input)
                encoder_end_event.record()
                # 2. Decoder 구간
                decoded_features = model.decoder(encoded_features)
                decoder_end_event.record()
                # 3. Classifier 구간
                _ = model.classifier(decoded_features)
                classifier_end_event.record()

                # 모든 이벤트가 기록된 후 동기화
                torch.cuda.synchronize() 
                total_times['encoder'] += start_event.elapsed_time(encoder_end_event)
                total_times['decoder'] += encoder_end_event.elapsed_time(decoder_end_event)
                total_times['classifier'] += decoder_end_event.elapsed_time(classifier_end_event)
                total_times['total'] += start_event.elapsed_time(classifier_end_event)
        
        avg_inference_time_per_sample = total_times['total'] / num_iterations
            
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        logging.info(f"샘플 당 평균 Forward Pass 시간: {avg_inference_time_per_sample:.2f}ms ({num_iterations}회 반복)")
        logging.info(f"  - Encoder: {total_times['encoder'] / num_iterations:.2f}ms")
        logging.info(f"  - Decoder: {total_times['decoder'] / num_iterations:.2f}ms")
        logging.info(f"  - Classifier: {total_times['classifier'] / num_iterations:.2f}ms")
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

    # 2. 테스트셋 성능 평가
    logging.info("="*50)
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
        eval_results = evaluate(run_cfg, model, data_loader, device, desc=f"[{mode_name}]", class_names=class_names, log_class_metrics=True)
        final_acc = eval_results['accuracy']

        # 3. 혼동 행렬 생성 및 저장 (최종 평가 시에만)
        if eval_results['labels'] and eval_results['preds']:
            plot_and_save_confusion_matrix(eval_results['labels'], eval_results['preds'], class_names, run_dir_path, timestamp)

    # 4. 어텐션 맵 시각화 (설정이 True인 경우)
    if model_cfg.save_attention:
        try:
            # 1. 어텐션 맵을 저장할 전용 폴더 생성
            attn_save_dir = os.path.join(run_dir_path, f'attention_map_{timestamp}')
            os.makedirs(attn_save_dir, exist_ok=True)

            num_to_save = min(getattr(model_cfg, 'num_plot_attention', 10), len(data_loader.dataset))
            logging.info(f"어텐션 맵 시각화를 시작합니다 ({num_to_save}개 샘플, 저장 위치: '{attn_save_dir}').")

            saved_count = 0 # 어텐션 맵을 저장할 전용 폴더 생성
            # 데이터 로더를 순회하며 num_to_save 개수만큼 시각화
            for sample_images, sample_labels, sample_filenames in data_loader:
                if saved_count >= num_to_save:
                    break

                sample_images = sample_images.to(device)
                batch_size = sample_images.size(0)

                # 모델을 실행하여 어텐션 맵이 저장되도록 함
                with torch.no_grad():
                    outputs = model(sample_images)

                _, predicted_indices = torch.max(outputs.data, 1)
                attention_maps = model.decoder.embedding4decoder.decoder.layers[-1].attn

                # 현재 배치에서 저장해야 할 샘플 수만큼 반복
                for i in range(batch_size):
                    if saved_count >= num_to_save:
                        break

                    predicted_class = class_names[predicted_indices[i].item()]
                    original_filename = sample_filenames[i]
                    
                    actual_class = "Unknown" if only_inference_mode else class_names[sample_labels[i].item()]

                    plot_and_save_attention_maps(
                        attention_maps, sample_images, attn_save_dir, model_cfg.img_size, model_cfg,
                        sample_idx=i, original_filename=original_filename, actual_class=actual_class, predicted_class=predicted_class
                    )
                    saved_count += 1
            
            logging.info(f"어텐션 맵 {saved_count}개 저장 완료.")
        except Exception as e:
            logging.error(f"어텐션 맵 시각화 중 오류 발생: {e}")
    return final_acc

def main():
    """메인 실행 함수"""
    # --- YAML 설정 파일 로드 --- #
    parser = argparse.ArgumentParser(description="YAML 설정을 이용한 이미지 분류기")
    parser.add_argument('--config', type=str, default='config.yaml', help='설정 파일 경로')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # SimpleNamespace를 사용하여 딕셔너리처럼 접근 가능하게 변환
    run_cfg = SimpleNamespace(**config['run'])
    train_cfg = SimpleNamespace(**config['training_run'])
    model_cfg = SimpleNamespace(**config['model'])
    # dataset_cfg도 SimpleNamespace로 변환
    run_cfg.dataset = SimpleNamespace(**run_cfg.dataset)
    
    data_dir_name = run_cfg.dataset.name

    # --- 실행 디렉토리 설정 ---
    if run_cfg.mode == 'train':
        # 훈련 모드: 새로운 실행 디렉토리 생성
        run_dir_path, timestamp = setup_logging(run_cfg, data_dir_name) # 여기서 run_dir_path와 timestamp가 반환됨
    elif run_cfg.mode == 'inference':
        # 추론 모드: 지정된 실행 디렉토리 사용
        run_dir_path = getattr(run_cfg, 'only_inference_dir', None)
        if getattr(run_cfg, 'show_log', True) and (not run_dir_path or not os.path.isdir(run_dir_path)):
            logging.error("추론 모드에서는 'config.yaml'에 'only_inference_dir'를 올바르게 설정해야 합니다.")
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
    use_cuda_if_available = getattr(run_cfg, 'cuda', True)
    device = torch.device("cuda" if use_cuda_if_available and torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        logging.info(f"CUDA 사용 가능. GPU 사용을 시작합니다. (Device: {torch.cuda.get_device_name(0)})")
    else:
        if use_cuda_if_available:
            logging.warning("config.yaml에서 CUDA 사용이 활성화되었지만, 사용 가능한 CUDA 장치를 찾을 수 없습니다. CPU를 사용합니다.")
        logging.info("CPU 사용을 시작합니다.")

    # --- 데이터 준비 ---
    train_loader, valid_loader, test_loader, num_labels, class_names = prepare_data(run_cfg, train_cfg, model_cfg)

    # --- 모델 구성 ---
    num_encoder_patches = (model_cfg.img_size // model_cfg.patch_size) ** 2 # 16
    
    decoder_params = {
        'num_encoder_patches': num_encoder_patches,
        'num_labels': num_labels,
        'num_decoder_layers': model_cfg.num_decoder_layers,
        'num_decoder_patches': model_cfg.num_decoder_patches,
        'featured_patch_dim': model_cfg.featured_patch_dim,
        'attn_pooling': getattr(model_cfg, 'attn_pooling', False),
        'emb_dim': model_cfg.emb_dim,
        'num_heads': model_cfg.num_heads,
        'decoder_ff_ratio': model_cfg.decoder_ff_ratio,
        'dropout': model_cfg.dropout,
        'positional_encoding': model_cfg.positional_encoding,
        'res_attention': model_cfg.res_attention,
        'save_attention': model_cfg.save_attention,
    }
    decoder_args = SimpleNamespace(**decoder_params)

    encoder = PatchConvEncoder(in_channels=model_cfg.in_channels, img_size=model_cfg.img_size, patch_size=model_cfg.patch_size, 
                               featured_patch_dim=model_cfg.featured_patch_dim, cnn_feature_extractor_name=model_cfg.cnn_feature_extractor['name'],
                               pre_trained=train_cfg.pre_trained)
    decoder = DecoderBackbone(args=decoder_args) # models.py의 Model 클래스
    
    classifier = Classifier(num_decoder_patches=model_cfg.num_decoder_patches,
                            featured_patch_dim=model_cfg.featured_patch_dim, num_labels=num_labels, dropout=model_cfg.dropout)
    model = HybridModel(encoder, decoder, classifier).to(device)

    # 모델 생성 후 파라미터 수 로깅
    log_model_parameters(model)
    
    # --- 모드에 따라 실행 ---
    if run_cfg.mode == 'train':
        optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr)
        scheduler = None

        scheduler_name = getattr(train_cfg, 'scheduler', 'none').lower()
        if scheduler_name == 'cosineannealinglr':
            logging.info(f"옵티마이저: AdamW, 스케줄러: CosineAnnealingLR (T_max={train_cfg.epochs})")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
        else:
            logging.info("옵티마이저: AdamW (스케줄러 없음)")

        logging.info("="*50)

        # 훈련 시에는 train_loader와 valid_loader 사용
        train(run_cfg, train_cfg, model, optimizer, scheduler, train_loader, valid_loader, device, run_dir_path, class_names)

        logging.info("="*50)
        logging.info("훈련 완료. 최종 모델 성능을 테스트 세트로 평가합니다.")
        final_acc = inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Test", class_names=class_names)

        # --- 그래프 생성 ---
        # 로그 파일 이름은 setup_logging에서 생성된 패턴을 기반으로 함
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
        # 추론 모드에서는 test_loader를 사용해 성능 평가
        inference(run_cfg, model_cfg, model, test_loader, device, run_dir_path, timestamp, mode_name="Inference", class_names=class_names)

# =============================================================================
# 5. 메인 실행 블록
# =============================================================================
if __name__ == '__main__':
    main()