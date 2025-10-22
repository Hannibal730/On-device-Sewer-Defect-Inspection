import re
import matplotlib.pyplot as plt
import argparse
import os

def visualize_log(log_file_path):
    """
    로그 파일에서 에포크별 [Valid] Accuracy를 추출하여 그래프를 그리고,
    화면에 표시합니다.

    Args:
        log_file_path (str): 분석할 로그 파일의 경로.
    """
    if not os.path.exists(log_file_path):
        print(f"오류: 로그 파일 '{log_file_path}'를 찾을 수 없습니다.")
        return

    epochs = []
    accuracies = []

    # run.py의 로그 형식에 맞는 정규 표현식
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
            print("로그 파일에서 유효한 [Valid] Accuracy 데이터를 찾을 수 없어 그래프를 생성할 수 없습니다.")
            return

        plt.style.use('seaborn-v0_8-whitegrid') # 그래프 스타일 설정
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, accuracies, marker='o', linestyle='-', color='dodgerblue', label='Validation Accuracy')
        
        plt.title(f'Validation Accuracy per Epoch\n(Source: {os.path.basename(log_file_path)})', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(70, 100)
        plt.xticks(epochs)
        plt.legend()
        
        # 그래프를 화면에 표시
        plt.show()

    except Exception as e:
        print(f"그래프 생성 중 오류 발생: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="로그 파일에서 Validation Accuracy를 시각화합니다.")
    # 기본 로그 파일 이름을 'viz.log'로 설정
    parser.add_argument('--log_file', type=str, default='viz.log', help='분석할 로그 파일 경로')
    args = parser.parse_args()

    visualize_log(args.log_file)