import os
import logging
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms, datasets

# =============================================================================
# 1. 데이터셋 클래스 정의
# =============================================================================
class CustomImageDataset(Dataset):
    """CSV 파일과 이미지 폴더 경로를 받아 데이터를 로드하는 커스텀 데이터셋입니다."""
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # 클래스 이름과 인덱스를 매핑합니다. (ImageFolder와 호환)
        self.classes = ['Normal', 'Defect']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

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

class ImageFolderWithPaths(datasets.ImageFolder):
    """기존 ImageFolder에 파일 경로(파일명)를 함께 반환하는 기능을 추가한 클래스입니다."""
    def __getitem__(self, index):
        # 기존 ImageFolder의 __getitem__을 호출하여 이미지와 레이블을 가져옵니다.
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # 파일 경로를 가져옵니다.
        path = self.imgs[index][0]
        return (*original_tuple, os.path.basename(path))

# =============================================================================
# 2. 데이터 준비 함수
# =============================================================================
def prepare_data(run_cfg, train_cfg, model_cfg):
    """데이터셋을 로드하고 전처리하여 DataLoader를 생성합니다."""
    img_size = model_cfg.img_size
    dataset_cfg = run_cfg.dataset

    # --- 데이터 샘플링 로직 함수 ---
    def get_subset(dataset, name, sampling_ratios, random_seed):
        """데이터셋에서 지정된 비율만큼 단순 랜덤 샘플링을 수행합니다."""
        ratio = 1.0
        if isinstance(sampling_ratios, dict):
            ratio = sampling_ratios.get(name, 1.0)
        elif isinstance(sampling_ratios, (float, int)):
            ratio = sampling_ratios

        if ratio < 1.0:
            logging.info(f"'{name}' 데이터셋을 {ratio * 100:.1f}% 비율로 샘플링합니다 (random_seed={random_seed}).")
            num_total = len(dataset)
            num_to_sample = int(num_total * ratio)
            num_to_sample = max(1, num_to_sample)
            rng = np.random.default_rng(random_seed)
            indices = rng.choice(num_total, size=num_to_sample, replace=False)
            return Subset(dataset, indices)
        return dataset

    # --- 데이터 변환(Transform) 정의 ---
    if model_cfg.in_channels == 1:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
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
    elif model_cfg.in_channels == 3:
        normalize = transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
        train_transform = transforms.Compose([
            # Sewer-ML 원본 논문은 Resize 대신에 RandomResizedCrop 을 사용했다.
            # transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)), 
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
        logging.info(f"'{dataset_cfg.name}' 데이터 로드를 시작합니다 (Type: {dataset_cfg.type}).")
        
        # --- 데이터셋 로드 ---
        if dataset_cfg.type == 'csv':
            full_train_dataset = CustomImageDataset(csv_file=dataset_cfg.paths['train_csv'], img_dir=dataset_cfg.paths['train_img_dir'], transform=train_transform)
            full_valid_dataset = CustomImageDataset(csv_file=dataset_cfg.paths['valid_csv'], img_dir=dataset_cfg.paths['valid_img_dir'], transform=valid_test_transform)
            full_test_dataset = CustomImageDataset(csv_file=dataset_cfg.paths['test_csv'], img_dir=dataset_cfg.paths['test_img_dir'], transform=valid_test_transform)
            class_names = full_train_dataset.classes
        elif dataset_cfg.type == 'image_folder':
            logging.info(f"'{dataset_cfg.paths['img_folder']}' 경로에서 데이터를 불러와 훈련/테스트셋으로 분할합니다.")
            
            # 훈련용(증강 포함)과 검증용(증강 없음) 데이터셋을 별도로 생성
            dataset_for_train = ImageFolderWithPaths(root=dataset_cfg.paths['img_folder'], transform=train_transform)
            dataset_for_valid_test = ImageFolderWithPaths(root=dataset_cfg.paths['img_folder'], transform=valid_test_transform)

            num_total = len(dataset_for_train)
            train_ratio = getattr(dataset_cfg, 'train_split_ratio', 0.8)
            num_train = int(num_total * train_ratio)
            num_test = num_total - num_train

            # 데이터 분할 시 사용할 시드를 config에서 가져옵니다. 없으면 기존 random_sampling_seed를 사용합니다.
            split_seed = getattr(dataset_cfg, 'split_seed', run_cfg.random_sampling_seed)
            logging.info(f"총 {num_total}개 데이터를 훈련용 {num_train}개, 테스트용 {num_test}개로 분할합니다 (split_seed={split_seed}).")

            # 재현성을 위해 고정된 시드로 데이터를 분할
            generator = torch.Generator().manual_seed(split_seed)
            train_indices, test_indices = random_split(range(num_total), [num_train, num_test], generator=generator)

            # 동일한 인덱스를 사용하여 각기 다른 transform을 가진 데이터셋의 Subset을 생성
            full_train_dataset = Subset(dataset_for_train, train_indices)
            full_test_dataset = Subset(dataset_for_valid_test, test_indices)
            full_valid_dataset = full_test_dataset # 검증셋은 테스트셋과 동일하게 사용
            
            class_names = dataset_for_train.classes
        else:
            raise ValueError(f"지원하지 않는 데이터셋 타입입니다: {dataset_cfg.type}")

        num_labels = len(class_names)
        logging.info(f"데이터셋 클래스: {class_names} (총 {num_labels}개)")

        # --- 데이터 샘플링 ---
        sampling_ratios = getattr(run_cfg, 'random_sampling_ratio', None)
        train_dataset = get_subset(full_train_dataset, 'train', sampling_ratios, run_cfg.random_sampling_seed)
        valid_dataset = get_subset(full_valid_dataset, 'valid', sampling_ratios, run_cfg.random_sampling_seed)
        test_dataset = get_subset(full_test_dataset, 'test', sampling_ratios, run_cfg.random_sampling_seed)

        # --- DataLoader 생성 ---
        # ImageFolder는 (image, label)을 반환하므로, CustomImageDataset과 형식을 맞추기 위해 collate_fn을 사용합니다.
        def collate_fn(batch):
            # 모든 데이터셋 클래스가 (image, label, filename) 튜플을 반환하도록 통일되었습니다.
            images, labels, filenames = zip(*batch)
            return torch.stack(images, 0), torch.tensor(labels), list(filenames)

        train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=run_cfg.num_workers, pin_memory=True, persistent_workers=True if run_cfg.num_workers > 0 else False, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=run_cfg.num_workers, pin_memory=True, persistent_workers=True if run_cfg.num_workers > 0 else False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=run_cfg.num_workers, pin_memory=True, persistent_workers=True if run_cfg.num_workers > 0 else False, collate_fn=collate_fn)
        
        logging.info(f"훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(valid_dataset)}개, 테스트 데이터: {len(test_dataset)}개")
        
        return train_loader, valid_loader, test_loader, num_labels, class_names
        
    except FileNotFoundError as e:
        logging.error(f"데이터 폴더 또는 CSV 파일을 찾을 수 없습니다: {e}. 'config.yaml'의 경로 설정을 확인해주세요.")
        exit()
    except Exception as e:
        logging.error(f"데이터 준비 중 오류 발생: {e}")
        exit()