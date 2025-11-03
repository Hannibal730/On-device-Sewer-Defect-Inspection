import os
import numpy as np
from argparse import ArgumentParser
from torchvision import models as torch_models
from torchvision import transforms
from collections import OrderedDict
import pandas as pd
import torch
import time
import logging
from datetime import datetime

try:
    from thop import profile
except ImportError:
    profile = None

from dataloader import MultiLabelDatasetInference
from torch.utils.data import DataLoader

import torch.nn as nn

import sewer_models
import ml_models


TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__ if name.islower() and not name.startswith("__") and callable(torch_models.__dict__[name]))
SEWER_MODEL_NAMES = sorted(name for name in sewer_models.__dict__ if name.islower() and not name.startswith("__") and callable(sewer_models.__dict__[name]))
MULTILABEL_MODEL_NAMES = sorted(name for name in ml_models.__dict__ if name.islower() and not name.startswith("__") and callable(ml_models.__dict__[name]))
MODEL_NAMES =  TORCHVISION_MODEL_NAMES + SEWER_MODEL_NAMES + MULTILABEL_MODEL_NAMES


def evaluate(dataloader, model, device):
    model.eval()

    all_sigmoid_predictions = None
    all_img_paths = []
    total_forward_time = 0.0
    sigmoid = nn.Sigmoid()

    num_batches = len(dataloader)
    
    with torch.no_grad():
        for i, (images, img_paths) in enumerate(dataloader):
            if i % 100 == 0:
                logging.info(f"Evaluating... {i} / {num_batches}")

            images = images.to(device)

            # --- 순수 forward pass 시간 측정 ---
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                output = model(images)
                end_event.record()

                torch.cuda.synchronize()
                total_forward_time += start_event.elapsed_time(end_event) / 1000.0
            else:
                start_time = time.time()
                output = model(images)
                end_time = time.time()
                total_forward_time += (end_time - start_time)

            sigmoid_output = sigmoid(output).detach().cpu().numpy()

            if all_sigmoid_predictions is None:
                all_sigmoid_predictions = sigmoid_output
            else:
                all_sigmoid_predictions = np.vstack((all_sigmoid_predictions, sigmoid_output))

            all_img_paths.extend(list(img_paths))
            
    return all_sigmoid_predictions, all_img_paths, total_forward_time


def load_model(model_path, best_weights=False):

    if best_weights:
        if not os.path.isfile(model_path):
            raise ValueError("The provided path does not lead to a valid file: {}".format(model_path))
        last_ckpt_path = model_path
    else:
        last_ckpt_path = os.path.join(model_path, "last.ckpt")
        if not os.path.isfile(last_ckpt_path):
            raise ValueError("The provided directory path does not contain a 'last.ckpt' file: {}".format(model_path))
    
    model_last_ckpt = torch.load(last_ckpt_path)
    

    model_name = model_last_ckpt["hyper_parameters"]["model"]
    num_classes = model_last_ckpt["hyper_parameters"]["num_classes"]
    training_mode = model_last_ckpt["hyper_parameters"]["training_mode"]
    br_defect = model_last_ckpt["hyper_parameters"]["br_defect"]
    
    # Load best checkpoint
    best_model = model_last_ckpt
    # if best_weights:
    #     best_model = model_last_ckpt
    # else:
    #     best_model_path = model_last_ckpt["checkpoint_callback_best_model_path"]
    #     best_model = torch.load(best_model_path)

    best_model_state_dict = best_model["state_dict"]

    updated_state_dict = OrderedDict()
    for k,v in best_model_state_dict.items():
        name = k.replace("model.", "")
        if "criterion" in name:
            continue

        updated_state_dict[name] = v

    return updated_state_dict, model_name, num_classes, training_mode, br_defect

def log_model_parameters(model):
    """모델의 총 파라미터 수를 계산하고 로깅합니다."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("="*50)
    logging.info("모델 파라미터 수:")
    logging.info(f"  - 총 파라미터: {total_params:,} 개")
    logging.info(f"  - 학습 가능한 파라미터: {trainable_params:,} 개")
    logging.info("="*50)


def setup_logging(log_dir, model_version, split):
    """로그 파일을 생성하고, 콘솔과 파일에 함께 출력하도록 설정합니다."""
    # 핸들러가 중복 추가되는 것을 방지하기 위해 기존 핸들러를 제거합니다.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"inference_{model_version}_{split.lower()}_{timestamp}.log")

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

def run_inference(args):
    
    ann_root = args["ann_root"]
    data_root = args["data_root"]
    model_path = args["model_path"]
    outputPath = args["results_output"]
    best_weights = args["best_weights"]
    # best_weights = False
    split = args["split"]
    
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
  
    updated_state_dict, model_name, num_classes, training_mode, br_defect = load_model(model_path, best_weights)

    if "model_version" not in args.keys():
        model_version = model_name
    else:
        model_version = args["model_version"]

    # --- 로깅 설정 ---
    setup_logging(outputPath, model_version, split)

    # Init model
    if model_name in TORCHVISION_MODEL_NAMES:
        model = torch_models.__dict__[model_name](num_classes = num_classes)
    elif model_name in SEWER_MODEL_NAMES:
        model = sewer_models.__dict__[model_name](num_classes = num_classes)
    elif model_name in MULTILABEL_MODEL_NAMES:
        model = ml_models.__dict__[model_name](num_classes = num_classes)
    else:
        raise ValueError("Got model {}, but no such model is in this codebase".format(model_name))

    model.load_state_dict(updated_state_dict)

    # 모델 파라미터 수 로깅
    log_model_parameters(model)
    
    # initialize dataloaders
    img_size = 299 if model in ["inception_v3", "chen2018_multilabel"] else 224
    
    eval_transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
        ])

    # split 값에 따라 올바른 이미지 폴더 이름을 결정합니다.
    if split == "Val":
        folder_name = "valid"
    else:
        folder_name = split.lower()
    img_dir = os.path.join(data_root, folder_name)
    dataset = MultiLabelDatasetInference(ann_root, img_dir, split=split, transform=eval_transform, onlyDefects=False)
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], num_workers = args["workers"], pin_memory=True)

    if training_mode in ["e2e", "defect"]:
        labelNames = dataset.LabelNames
    elif training_mode == "binary":
        labelNames = ["Defect"]
    elif training_mode == "binaryrelevance":
        labelNames = [br_defect]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # 1. GPU 메모리 사용량 측정
    # 모델의 입력 크기를 확인하기 위해 샘플 이미지를 하나 가져옵니다.
    sample_image, _ = dataset[0]
    dummy_input = sample_image.unsqueeze(0).to(device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        with torch.no_grad():
            _ = model(dummy_input)
            
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        logging.info(f"샘플 당 Forward Pass 시 최대 GPU 메모리 사용량: {peak_memory_mb:.2f} MB")
    else:
        logging.info("CUDA를 사용할 수 없어 GPU 메모리 사용량을 측정하지 않습니다.")

    # 1-2. FLOPs (연산량) 측정
    gflops_per_sample = 0.0
    if profile:
        # thop.profile은 MACs를 반환합니다. FLOPS는 보통 MACs * 2 입니다.
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        # GFLOPS (Giga Floating Point Operations) 단위로 변환
        gflops_per_sample = (macs * 2) / 1e9
        logging.info(f"연산량 (FLOPs): {gflops_per_sample:.2f} GFLOPs per sample")
    else:
        logging.info("연산량 (FLOPs): N/A (thop 라이브러리가 설치되지 않아 측정을 건너뜁니다.)")
        logging.info("  - FLOPs를 측정하려면 'pip install thop'을 실행하세요.")


    # 2. 추론 및 성능 평가
    logging.info(f"{split} 데이터셋에 대한 추론을 시작합니다...")
    sigmoid_predictions, val_imgPaths, total_forward_time = evaluate(dataloader, model, device)

    num_test_samples = len(dataloader.dataset)
    avg_inference_time_per_sample = (total_forward_time / num_test_samples) * 1000 if num_test_samples > 0 else 0

    logging.info(f"총 Forward Pass 시간: {total_forward_time:.2f}s (테스트 샘플 {num_test_samples}개)")
    logging.info(f"샘플 당 평균 Forward Pass 시간: {avg_inference_time_per_sample:.2f}ms")

    sigmoid_dict = {}
    sigmoid_dict["Filename"] = val_imgPaths
    for idx, header in enumerate(labelNames):
        sigmoid_dict[header] = sigmoid_predictions[:,idx]
    sigmoid_df = pd.DataFrame(sigmoid_dict)
    result_csv_path = os.path.join(outputPath, "{}_{}_sigmoid.csv".format(model_version, split.lower()))
    sigmoid_df.to_csv(result_csv_path, sep=",", index=False)
    logging.info(f"추론 결과가 '{result_csv_path}'에 저장되었습니다.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='pytorch_gpu')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default="/home/user/workspace/disk/nvme1/data/Sewer/Sewer-ML")
    parser.add_argument('--data_root', type=str, default="/home/user/workspace/disk/nvme1/data/Sewer/Sewer-ML")
    parser.add_argument('--batch_size', type=int, default=64, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument("--model_path", type=str, default="/home/user/workspace/CHOI/Multi-label-Sewer-Classification/pretrained_models/xie2019_binary-binary-version_1.pth")
    parser.add_argument("--best_weights", action="store_true", help="If true 'model_path' leads to a specific weight file. If False it leads to the output folder of lightning_trainer where the last.ckpt file is used to read the best model weights.")
    parser.add_argument("--results_output", type=str, default = "./inference_results")
    parser.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])



    args = vars(parser.parse_args())

    # best_weights 인자를 True로 설정하여 실행
    args['best_weights'] = True
    run_inference(args)