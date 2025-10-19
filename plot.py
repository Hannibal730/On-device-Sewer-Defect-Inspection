import os
import re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_and_save_accuracy_graph(log_file_path, save_dir, final_acc):
    """
    로그 파일에서 에포크별 [Valid] Accuracy를 추출하여 그래프를 그리고,
    지정된 디렉토리에 이미지 파일로 저장합니다.

    Args:
        log_file_path (str): 분석할 로그 파일의 전체 경로.
        save_dir (str): 그래프 이미지를 저장할 디렉토리 경로.
        final_acc (float): 그래프 제목에 표시할 최종 추론 정확도.
    """
    epochs = []
    accuracies = []

    # 정규 표현식을 사용하여 로그 라인에서 에포크와 정확도 추출
    # 예: [Valid] [1/30] | Val Acc: 85.00% ...
    pattern = re.compile(r"\[Valid\] \[(\d+)/\d+\] \| Val Acc: ([\d\.]+)%")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epochs.append(int(match.group(1)))
                    accuracies.append(float(match.group(2)))

        if not epochs:
            logging.warning("로그 파일에서 유효한 [Valid] Accuracy 데이터를 찾을 수 없어 그래프를 생성하지 않습니다.")
            return

        plt.figure(figsize=(15, 12))
        plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b')
        plt.title(f'Validation Accuracy per Epoch (Final Test Acc: {final_acc:.2f}%)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100) # Y축 범위를 0부터 100까지로 고정
        plt.grid(True)
        plt.xticks(epochs) # 모든 에포크 번호를 x축에 표시

        save_path = os.path.join(save_dir, 'validation_accuracy_plot.png')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Val Acc 그래프가 '{save_path}'에 저장되었습니다.")

    except FileNotFoundError:
        logging.error(f"로그 파일 '{log_file_path}'를 찾을 수 없습니다.")
    except Exception as e:
        logging.error(f"그래프 생성 중 오류 발생: {e}")

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    실제 값과 예측 값을 기반으로 혼동 행렬(Confusion Matrix)을 계산하고,
    이를 시각화하여 지정된 경로에 이미지 파일로 저장합니다.

    Args:
        y_true (list or np.array): 실제 레이블 리스트.
        y_pred (list or np.array): 모델이 예측한 레이블 리스트.
        class_names (list of str): 클래스 이름 리스트 (e.g., ['Normal', 'Defect']).
        save_path (str): 혼동 행렬 이미지를 저장할 전체 경로.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 16}) # 숫자 크기 조절
        
        plt.title('Confusion Matrix', fontsize=20)
        plt.ylabel('Actual', fontsize=15)
        plt.xlabel('Predicted', fontsize=15)
        
        plt.savefig(save_path)
        plt.close()
        logging.info(f"혼동 행렬 그래프가 '{save_path}'에 저장되었습니다.")
    except Exception as e:
        logging.error(f"혼동 행렬 그래프 생성 중 오류 발생: {e}")