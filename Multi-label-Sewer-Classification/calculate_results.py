import os
import json
import argparse
import pandas as pd
from metrics import evaluation
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


MultiLabelWeightDict = {"RB":1.00,"OB":0.5518,"PF":0.2896,"DE":0.1622,"FS":0.6419,"IS":0.1847,"RO":0.3559,"IN":0.3131,"AF":0.0811,"BE":0.2275,"FO":0.2477,"GR":0.0901,"PH":0.4167,"PB":0.4167,"OS":0.9009,"OP":0.3829,"OK":0.4396}
MultiLabels = list(MultiLabelWeightDict.keys())
LabelWeights = list(MultiLabelWeightDict.values())

def calculateResults(args):
    def plot_and_save_confusion_matrix(y_true, y_pred, class_names, save_path):
        """
        혼동 행렬을 계산하고 시각화하여 이미지 파일로 저장합니다.
        이진 분류의 경우에만 작동합니다.
        """
        if not class_names:
            print("클래스 이름이 없어 혼동 행렬을 생성할 수 없습니다.")
            return

        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names,
                        annot_kws={"size": 16})
            plt.title('Confusion Matrix', fontsize=20)
            plt.ylabel('Actual', fontsize=15)
            plt.xlabel('Predicted', fontsize=15)
            plt.savefig(save_path)
            plt.close()
            print(f"혼동 행렬 저장 완료: '{save_path}'")
        except Exception as e:
            print(f"혼동 행렬 생성 중 오류 발생: {e}")

    scorePath = args["score_path"]
    targetPath = args["gt_path"]

    outputPath = args["output_path"]

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    split = args["split"]

    targetSplitpath = os.path.join(targetPath, "SewerML_{}.csv".format(split))
    targetsDf = pd.read_csv(targetSplitpath, sep=",")

    for subdir, dirs, files in os.walk(scorePath):
        
        for scoreFile in files:
            item = os.path.splitext(scoreFile)
            
            if split.lower() not in item[0]:
                continue
            # if "e2e" not in item[0] and "twostage" not in item[0] and "defect" not in item[0]:
            #     continue
            if "sigmoid" not in item[0]:
                continue
            if item[1] != ".csv":  # if os.path.splitext(scoreFile)[-1] != ".csv":
                continue
            

            scoresDf = pd.read_csv(os.path.join(subdir, scoreFile), sep=",")
            scoresDf = scoresDf.sort_values(by=["Filename"]).reset_index(drop=True)

            # 파일 이름에 'binary'가 포함되어 있는지 확인하여 레이블 목록을 결정합니다.
            if "binary" in item[0]:
                Labels = ["Defect"]
                LabelWeights = [1.0] # 이진 분류이므로 가중치는 1.0으로 설정
            else:
                Labels = MultiLabels
                LabelWeights = list(MultiLabelWeightDict.values())
            
            # 정렬된 scoresDf와 일치하도록 targetsDf도 정렬합니다.
            current_targetsDf = targetsDf.sort_values(by=["Filename"]).reset_index(drop=True)
            targets = current_targetsDf[Labels].values
            scores = scoresDf[Labels].values
            
            # --- 혼동 행렬 생성 및 저장 (이진 분류의 경우) ---
            if "binary" in item[0]:
                # scores는 확률값이므로, 0.5를 기준으로 0 또는 1로 변환
                predictions = (scores >= 0.5).astype(int).flatten()
                ground_truth = targets.flatten()
                
                # 클래스 이름: [정상, 결함]
                binary_class_names = ['Normal', 'Defect']
                
                # 파일명 설정
                cm_filename = scoreFile.replace(f"_{split.lower()}_sigmoid.csv", "_confusion_matrix.png")
                cm_save_path = os.path.join(outputPath, cm_filename)
                plot_and_save_confusion_matrix(ground_truth, predictions, binary_class_names, cm_save_path)

            new, main, auxillary = evaluation(scores, targets, LabelWeights)

            # scoreFile 이름에서 '_<split>_sigmoid.csv' 부분을 제거하여 outputName을 생성합니다.
            # 예: 'xie2019_binary_val_sigmoid.csv' -> 'xie2019_binary'
            suffix_to_remove = f"_{split.lower()}_sigmoid.csv"
            outputName = scoreFile.replace(suffix_to_remove, "")

            with open(os.path.join(outputPath,'{}.json'.format(outputName)), 'w') as fp:
                json.dump({"Labels": Labels, "LabelWeights": LabelWeights, "New": new, "Main": main, "Auxillary": auxillary}, fp)

            # --- 상세한 결과 출력을 위한 문자열 생성 ---
            # 'Normal' 클래스는 항상 마지막에 위치합니다.
            class_names_with_normal = Labels + ["Normal"]

            def format_metric_line(metric_name, values, names):
                """지표 이름과 값, 클래스 이름을 결합하여 한 줄의 문자열로 만듭니다."""
                # AP는 'Normal' 클래스가 없으므로 이름 목록을 따로 받습니다.
                items = [f"{name}: {value*100:.4f}" for name, value in zip(names, values)]
                return f"{metric_name}: " + "   ".join(items)

            # New metrics
            newString = f"F2-CIW: {new['F2']*100:.4f}   F1-Normal: {auxillary['F1_class'][-1]*100:.4f}"

            # ML main metrics
            main_metric_names = ["mF1", "MF1", "OF1", "OP", "OR", "CF1", "CP", "CR", "EMAcc", "mAP"]
            main_metric_values = [main[name] for name in main_metric_names]
            aveargeString = "   ".join([f"{name}: {value*100:.4f}" for name, value in zip(main_metric_names, main_metric_values)])

            # Class-wise metrics
            classF1String = format_metric_line("Class F1", auxillary["F1_class"], class_names_with_normal)
            classF2String = format_metric_line("Class F2", new["F2_class"], class_names_with_normal)
            classPString = format_metric_line("Class Precision", auxillary["P_class"], class_names_with_normal)
            classRString = format_metric_line("Class Recall", auxillary["R_class"], class_names_with_normal)
            classAPString = format_metric_line("Class AP", auxillary["AP"], Labels) # AP는 Defect 클래스에 대해서만 계산됨

            with open(os.path.join(outputPath,'{}_latex.txt'.format(outputName)), "w") as text_file:
                text_file.write(newString + "\n")
                text_file.write(aveargeString + "\n\n")
                text_file.write(classF1String + "\n")
                text_file.write(classF2String + "\n")
                text_file.write(classPString + "\n")
                text_file.write(classRString + "\n")
                text_file.write(classAPString + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default = "./calculate_results")
    parser.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])
    parser.add_argument("--score_path", type=str, default = "./inference_results")
    parser.add_argument("--gt_path", type=str, default = "/home/user/workspace/disk/nvme1/data/Sewer/Sewer-ML")

    args = vars(parser.parse_args())

    calculateResults(args)