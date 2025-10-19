import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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