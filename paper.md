논문 제목: Low Cost Sewer Defect Diagnosis Transformer for On-Device Processing
저자 및 소속

최근 도시 인프라의 노후화로 인해 하수관의 주기적 점검 및 유지보수의 중요성이 급격히 증가하고 있다. 기존의 하수관 점검 시스템은 고성능 서버나 클라우드 환경에서 작동하도록 설계되어, 현장 실시간 분석 및 자동화 점검에 제약이 존재한다. 본 연구에서는 이러한 한계를 극복하기 위해 엣지 디바이스(On-Device) 상에서 효율적으로 동작하는 Cross-Attention Sharing 기반 경량화 모델을 제안한다. 제안 모델은 다중 시각 특징 맵 간의 Cross-Attention 정보를 공유함으로써, 하수관 내 결함의 특징 표현을 강화하면서도 연산량과 메모리 사용량을 크게 줄인다. 또한, Sewer-ML 및 자체 구축 하수관 영상 데이터셋을 활용하여 모델의 성능을 검증한 결과, 기존의 CNN 및 Transformer 기반 경량 모델 대비 25% 이상의 파라미터 절감과 함께 동등 이상의 탐지 정확도를 달성하였다. 본 연구는 하수관 점검 로봇 및 스마트 시티 인프라 관리 시스템에 적용 가능한 실시간, 저전력 하수관 결함 탐지 기술로서의 활용 가능성을 제시한다.


초록 (Abstract)

도시 기반 시설의 노후화로 인해 하수관의 정기적인 점검 및 유지보수의 중요성이 커지고 있습니다. Due to the aging of urban infrastructure, the importance of regular inspection and maintenance of sewer pipes is increasing.

기존 딥러닝 기반 결함 탐지 시스템은 고성능 서버나 클라우드 환경에 의존적이어서, 하수관 점검 로봇과 같은 엣지 디바이스에서의 실시간 처리 및 자동화 점검에 제약이 있습니다. Existing deep learning-based defect detection systems are dependent on high-performance servers or cloud environments, which limits real-time processing and automated inspection on edge devices such as sewer inspection robots.

본 연구에서는 이러한 한계를 해결하고자 **LC-Transformer(Low-Cost Transformer)**를 제안합니다. In this paper, we propose the LC-Transformer (Low-Cost Transformer) to address these limitations.

이는 On-device 하수관 결함 진단을 위해 특별히 설계된 새로운 경량 하이브리드 아키텍처입니다. It is a novel lightweight hybrid architecture specially designed for On-device sewer defect diagnosis.

제안 모델은 파라미터 효율성을 극대화하기 위해 인코더 단계에서 경량 CNN 가중치를 입력 이미지의 패치마다 공유합니다. To maximize parameter efficiency, the proposed model shares lightweight CNN weights across input image's patches in the encoder stage.

그리고 디코더 단계에서는 계산 비용을 크게 줄이기 위해 크로스-어텐션(cross-attention) 방식을 사용합니다. And in the decoder stage, it uses a cross-attention method to significantly reduce computational costs.

또한 디코더의 성능을 높이기 위해서 소수의 학습 가능한 쿼리로부터 이미지 특징에 적응적인 '동적 쿼리(dynamic query)'를 생성하는 어텐션 풀링(attention pooling) 방식을 도입했습니다. Furthermore, to enhance the decoder's performance, we introduced an attention pooling method that generates 'dynamic queries' adaptive to image features from a small number of learnable queries.

하수도 데이터셋을 이용한 실험 결과, 제안 모델은 기존 하수도 전용 모델과 경량 모델들보다 현저히 적은 파라미터와 연산비용으로도 동등하거나 더 우수한 진단 성능을 달성했습니다. Experimental results on the sewer dataset show that the proposed model achieves equivalent or superior diagnostic performance with significantly fewer parameters and computational cost than existing sewer-specific and lightweight models.

본 연구는 하수관 점검 로봇 및 스마트 시티 인프라 관리 시스템과 같은 On-device에 적용 가능한 실시간, 저전력 하수관 결함 탐지 기술로써의 활용 가능성을 제시합니다. This study presents the potential of our work as a real-time, low-power sewer defect detection technology applicable to On-device systems, such as sewer inspection robots and smart city infrastructure management systems.


----시작

Urban sewer infrastructure is aging, making routine inspection and maintenance increasingly critical. However, existing deep learning–based defect detectors typically depend on high-end servers or cloud back ends, limiting real-time, automated operation on edge platforms such as sewer-inspection robots. We introduce LC-Transformer (Low-Cost Transformer), a lightweight hybrid architecture for on-device sewer defect diagnosis. In the encoder, a compact CNN shares its weights across image patches, maximizing parameter efficiency; in the decoder, a cross-attention mechanism substantially reduces computation. To further enhance decoding, we employ attention pooling to derive dynamic queries—adapted to the input features—from a small set of learnable seed queries. On a sewer-inspection dataset, LC-Transformer matches or surpasses specialized sewer models and prior lightweight baselines while using significantly fewer parameters and compute. These results highlight LC-Transformer’s potential as a real-time, low-power solution deployable on inspection robots and smart-city infrastructure management systems.

----끝



1. 서론 (Introduction)
도시의 하수관 시스템은 위생 및 환경 유지에 필수적인 기반 시설입니다. The urban sewer system is an essential infrastructure for sanitation and environmental maintenance.

그러나 많은 도시에서 하수관의 노후화가 빠르게 진행되고 있으며, 이는 균열, 침입, 붕괴 등 심각한 결함으로 이어져 도로 함몰이나 환경 오염과 같은 사회적 문제를 야기할 수 있습니다. However, in many cities, sewer pipes are rapidly aging, which can lead to severe defects such as cracks, intrusions, and collapses, causing social problems like road sinkholes and environmental pollution.

따라서 CCTV 로봇을 활용한 주기적인 내부 점검이 필수적이지만, 방대한 양의 비디오 데이터를 전문가가 직접 육안으로 판독하는 전통적인 방식은 시간과 비용이 많이 소요될 뿐만 아니라, 판독자의 주관에 따라 일관성이 결여될 수 있습니다. Therefore, periodic internal inspections using CCTV robots are essential, but the traditional method of experts manually reviewing vast amounts of video data is not only time-consuming and costly but also prone to inconsistency due to the subjective judgment of the reviewer.

이러한 문제를 해결하기 위해 딥러닝 기반의 자동화된 결함 탐지 연구가 활발히 진행되어 왔습니다. To address these issues, research on deep learning-based automated defect detection has been actively conducted.

초기 연구들은 ResNet, VGG 등 고성능 CNN 모델을 활용하여 높은 탐지 정확도를 달성하는 데 집중했습니다. Early studies focused on achieving high detection accuracy using high-performance CNN models such as ResNet and VGG.

하지만 이러한 모델들은 막대한 연산량과 메모리를 요구하여 고성능 GPU 서버나 클라우드 환경에 의존적입니다. However, these models demand immense computational power and memory, making them dependent on high-performance GPU servers or cloud environments.

이는 현장에서 사용되는 하수관 점검 로봇과 같은 엣지 디바이스(On-device)에 탑재하기 어렵게 만드는 한계로 작용합니다. This poses a limitation, making them difficult to deploy on edge devices like sewer inspection robots used in the field.

본 논문에서는 이러한 한계를 극복하고, 자원이 제한된 엣지 환경에서 하수관 결함을 효율적으로 진단할 수 있는 **LC-Transformer(Low-Cost Transformer)**를 제안합니다. In this paper, we propose the LC-Transformer (Low-Cost Transformer) to overcome these limitations and efficiently diagnose sewer defects in resource-constrained edge environments.

LC-Transformer는 CNN의 효율적인 지역 특징 추출 능력과 트랜스포머 기반의 정보 공유 능력을 결합한 새로운 경량 하이브리드 아키텍처입니다. The LC-Transformer is a novel lightweight hybrid architecture that combines the efficient local feature extraction capabilities of CNNs with the information-sharing capabilities of Transformer-based models.

특히, 저희는 모델의 경량화와 효율성에 초점을 맞추어 두 가지 핵심 아이디어를 제시합니다. Specifically, we present two core ideas focused on model lightweighting and efficiency.

*   **파라미터 효율성을 극대화한 하이브리드 인코더-디코더 구조**: 이미지를 여러 패치로 나누고, 모든 패치가 경량 CNN 가중치를 공유하도록 패치 인코더를 설계하여 파라미터 수를 획기적으로 줄였습니다.
    *   **Parameter-Efficient Hybrid Encoder-Decoder Architecture**: We drastically reduced the number of parameters by designing a patch encoder that divides the image into multiple patches and shares lightweight CNN weights across all patches.
    또한, 디코더는 파라미터 및 연산 효율성을 높이기 위해 기존 트랜스포머 기반 모델과 달리 self-attention 레이어를 생략하고 cross-attention만을 사용하여 패치들의 특징을 종합하고 최종 예측을 수행합니다.
    Furthermore, to enhance parameter and computational efficiency, the decoder, unlike conventional Transformer-based models, omits the self-attention layer and uses only cross-attention to aggregate patch features and perform the final prediction.
*   **어텐션 풀링을 통한 동적 쿼리 생성**: 기존 트랜스포머 모델의 고정된 쿼리 대신, 소수의 학습 가능한 '잠재 쿼리'가 입력 이미지의 특징들에 적응 가능하도록 '동적 쿼리'를 생성하는 어텐션 풀링 메커니즘을 제안합니다.
    *   **Dynamic Query Generation via Attention Pooling**: Instead of the fixed queries of traditional Transformer models, we propose an attention pooling mechanism that generates 'dynamic queries' from a small number of learnable 'latent queries' to be adaptive to the features of the input image.
    이 방식은 각 이미지에 가장 중요한 정보에 집중할 수 있도록 도움과 동시에, 잠재 쿼리 이외에는 추가적인 파라미터를 필요로 하지 않기 때문에 디코더의 복잡도를 낮추기까지 합니다.
    This approach not only helps to focus on the most important information in each image but also reduces the decoder's complexity as it requires no additional parameters beyond the latent queries.

저희는 공개 데이터셋 Sewer-ML과 자체 구축 하수관 영상 데이터셋에 대한 실험을 통해, 제안 모델이 기존 경량 모델들 대비 분류 성능은 유지하거나 능가하면서도 파라미터 수, 연산량(FLOPs), 추론 시간, 메모리 사용량 등 온디바이스 환경의 핵심 지표에서 효율성을 보임을 입증했습니다.
Through experiments on the public Sewer-ML dataset and our self-constructed sewer video dataset, we have demonstrated that our proposed model shows efficiency in key on-device metrics such as the number of parameters, FLOPs, inference time, and memory usage, while maintaining or surpassing the classification performance of existing lightweight models.

본 논문의 나머지 부분은 다음과 같이 구성됩니다.
The remainder of this paper is organized as follows.
2장에서는 관련 연구를 소개하고, 3장에서는 제안하는 LC-Transformer의 구조를 상세히 설명합니다.
Section 2 introduces related work, and Section 3 details the proposed LC-Transformer architecture.
4장에서는 실험 설정 및 결과를 분석하며, 마지막 5장에서 결론을 맺습니다.
Section 4 analyzes the experimental setup and results, and finally, Section 5 concludes the paper.


---- 시작

Introduction

Urban sewer networks are critical infrastructure for public sanitation and environmental protection. However, many systems are aging rapidly, leading to defects such as cracks, intrusions, and collapses that can trigger road sinkholes and environmental pollution. Periodic internal inspections with CCTV robots are therefore essential, yet manual review of large video corpora is time-consuming, costly, and susceptible to inconsistency due to subjective judgments.

Deep learning–based automated defect detection has gained traction as a remedy. Early studies centered on high-capacity CNNs (e.g., VGG, ResNet) to maximize detection accuracy, but their heavy computational and memory demands make them reliant on high-end GPU servers or cloud resources, hindering deployment on edge devices such as fielded sewer-inspection robots.

We introduce LC-Transformer (Low-Cost Transformer), a lightweight hybrid architecture tailored for resource-constrained edge environments that combines the local feature efficiency of CNNs with the information-sharing benefits of Transformer-style attention. The design is built on two key ideas:

1. Parameter-efficient hybrid encoder–decoder. Images are partitioned into patches that share a single lightweight CNN patch-encoder, substantially reducing the parameter count. To further improve parameter and compute efficiency, the decoder omits self-attention and uses only cross-attention to aggregate patch features and produce the final prediction.

2. Dynamic query generation via attention pooling. Instead of fixed queries, a small set of learnable latent queries is adapted to each input through attention pooling, yielding input-conditioned dynamic queries. This focuses computation on the most informative content in each image while introducing no additional parameters beyond the latent queries, thereby lowering decoder complexity.

Experiments on the public Sewer-ML dataset and a private sewer-video dataset demonstrate that LC-Transformer matches or surpasses the classification performance of existing lightweight models while delivering superior on-device metrics, including fewer parameters, lower FLOPs, faster inference, and reduced memory usage.

The remainder of the paper is organized as follows. Section 2 reviews related work. Section 3 details the LC-Transformer architecture. Section 4 presents the experimental setup and results. Section 5 concludes.


----- 끝




2. 관련 연구 (Related Work)
2.1. 하수관 결함 탐지를 위한 모델
2.2. 경량 CNN 모델
2.3. 경량 트랜스포머 모델
2.4. 하이브리드 CNN-Transformer 모델




3. Methodology
3.1. 전체 아키텍처 개요

제안하는 LC-Transformer는 입력 이미지 $I$를 받아 최종 로짓(Logits)을 출력하는 함수 $f_{LC}$로 정의할 수 있습니다.
$$ \text{Logits} = f_{LC}(I; \theta) = (\Phi_{cls} \circ \Phi_{dec} \circ \Phi_{enc})(I) $$
여기서 $\Phi_{enc}$, $\Phi_{dec}$, $\Phi_{cls}$는 각각 **패치 인코더, 디코더, 분류기**를 나타내며, $\theta$는 모델의 모든 학습 가능한 파라미터입니다.

3.2. 패치 컨볼루션 인코더 (Patch Convolutional Encoder)

패치 인코더 $\Phi_{enc}$는 2D 이미지를 1D 특징 시퀀스(1D feature sequence)로 변환하는 역할을 합니다. 이 과정은 다음과 같은 세 단계로 구성됩니다.

1.  **입력 및 패치 분할 (Input and Patching)**: 입력 이미지 $I \in \mathbb{R}^{H \times W \times C}$는 서로 겹치지 않는 $N_p$개의 패치 $\{p_1, p_2, ..., p_{N_p}\}$로 분할됩니다. 각 패치 $p_i$의 크기는 $\mathbb{R}^{S \times S \times C}$이며, 여기서 $S$는 패치 크기(patch size)이고 총 패치 수 $N_p$는 $(H/S) \times (W/S)$입니다.

2.  **가중치 공유 기반 특징 추출 (Weight-Shared Feature Extraction)**: 모델의 파라미터 수를 효율적으로 줄이기 위해, 분할된 모든 패치($p_i$)는 **가중치를 공유하는 CNN 특징 추출기**($\Phi_{feat}$)를 통과합니다. 이는 각 패치마다 별도의 특징 추출기를 두는 대신 단 하나의 경량 CNN을 재사용하는 방식으로, 본 연구에서는 EfficientNet-B0의 초기 레이어를 $\Phi_{feat}$로 활용했습니다. 이후 각 패치에서 추출된 특징 맵은 AdaptiveAvgPool을 거쳐 고정된 크기로 조절되고, 최종적으로 $D_{feat}$ 차원의 특징 벡터 $\mathbf{f}_i$로 변환됩니다.
    $$ \mathbf{f}_i = \text{Flatten}(\text{AdaptiveAvgPool}(\text{LightweightCNN}(p_i))) \in \mathbb{R}^{D_{feat}} $$

3.  **특징 시퀀스 생성 (Feature Sequence Generation)**: 2단계에서 추출된 모든 패치 특징 벡터 $\{\mathbf{f}_1, \mathbf{f}_2, ..., \mathbf{f}_{N_p}\}$를 순차적으로 결합(concatenate)하고 Layer Normalization을 적용하여, 디코더로 전달될 최종 인코더 출력 시퀀스 $X_{enc} \in \mathbb{R}^{N_p \times D_{feat}}$를 생성합니다.
    $$ X_{enc} = \text{LayerNorm}([\mathbf{f}_1; \mathbf{f}_2; ...; \mathbf{f}_{N_p}]) $$


3.3. 경량 크로스-어텐션 디코더 (Lightweight Cross-Attention Decoder)

디코더 $\Phi_{dec}$는 인코더가 생성한 패치 특징 시퀀스 $X_{enc}$를 입력받아, 이미지 분류에 필요한 핵심 정보를 추출하고 요약하는 역할을 수행합니다.
3.3.1. 디코더 입력 준비 (Input Preparation)
디코더의 입력으로 사용될 키(Key)와 값(Value)은 인코더 출력 시퀀스 $X_{enc}$에 선형 변환 $W_{emb}$와 학습 가능한 위치 인코딩 $PE \in \mathbb{R}^{N_p \times D_{emb}}$를 적용하여 생성됩니다.
$$ K = V = W_{emb}(X_{enc}) + PE \in \mathbb{R}^{N_p \times D_{emb}} $$


3.3.2. 어텐션 풀링을 통한 동적 쿼리 생성 (Dynamic Query Generation via Attention Pooling)
어텐션 풀링은 입력 이미지의 특징에 적응적인 '동적 쿼리(dynamic query)'를 생성하여 디코더의 효율과 성능을 동시에 높입니다.

특히 이 과정은 일반적인 트랜스포머의 어텐션 연산과 달리, 쿼리, 키, 값을 위한 별도의 선형 변환 행렬($W_Q, W_K, W_V$)을 사용하지 않습니다. 대신, 인코더에서 넘어온 키(K)와 값(V) 시퀀스를 그대로 활용하여 잠재 쿼리와의 어텐션 스코어를 계산합니다. 그 결과, **어텐션 풀링 단계에서는 학습 가능한 '잠재 쿼리'($Q_{latent}$) 외에 추가적인 파라미터가 발생하지 않아 모델 경량화에 크게 기여**합니다.

이 과정은 다음과 같이 세 단계로 이루어집니다.

1.  **잠재 쿼리 준비**: 모델은 $N_q$개의 학습 가능한 잠재 쿼리 $Q_{latent} \in \mathbb{R}^{N_q \times D_{feat}}$를 보유합니다. 이 쿼리는 선형 변환을 거쳐 임베딩 차원으로 변환됩니다.
    $$ Q'_{latent} = W_{emb}(Q_{latent}) \in \mathbb{R}^{N_q \times D_{emb}} $$
2.  **어텐션 스코어 계산**: 변환된 잠재 쿼리 $Q'_{latent}$와 인코더 출력의 키(Key) 시퀀스 $K$ 사이의 어텐션 스코어가 계산됩니다.
    $$ \text{Scores} = \text{softmax}\left(\frac{Q'_{latent} K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{N_q \times N_p} $$
3.  **동적 쿼리 생성**: 계산된 어텐션 스코어를 값(Value) 시퀀스 $V$에 가중합하여 최종 동적 쿼리 $Q_{dynamic}$가 생성됩니다. 이 동적 쿼리는 디코더의 첫 번째 레이어에 입력으로 사용됩니다.
    $$ Q_{dynamic} = \text{Scores} \cdot V \in \mathbb{R}^{N_q \times D_{emb}} $$


3.3.3. 디코더 레이어 (Decoder Layer)
$L$개의 디코더 레이어는 Multi-Head Cross-Attention (MHCA)과 Feed-Forward Network (FFN)로 구성된다. $l$-번째 레이어의 입력 쿼리를 $Q^{(l-1)}$이라 할 때, 출력 $Q^{(l)}$은 다음과 같이 계산된다. $$ \hat{Q}^{(l)} = \text{LayerNorm}(Q^{(l-1)} + \text{MHCA}(Q^{(l-1)}, K, V)) $$ $$ Q^{(l)} = \text{LayerNorm}(\hat{Q}^{(l)} + \text{FFN}(\hat{Q}^{(l)})) $$
여기서 MHCA는 $h$개의 헤드를 사용하는 멀티-헤드 어텐션 메커니즘을 나타내며, 다음과 같이 정의됩니다.
$$ \text{MHCA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O $$
각 어텐션 헤드 $\text{head}_i$는 입력 쿼리 $Q$, 키 $K$, 값 $V$를 각각의 선형 변환 행렬 $W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{D_{emb} \times d_k}$를 사용하여 변환(projection)한 후, 스케일드 닷-프로덕트 어텐션(Scaled Dot-Product Attention)을 수행합니다.
$$ \text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i) $$
$$ \text{Attention}(Q', K', V') = \text{softmax}\left(\frac{Q'K'^T}{\sqrt{d_{k}}}\right)V' $$
이때, $d_{k} = D_{emb}/h$는 각 헤드의 차원을 나타내며, $W_O \in \mathbb{R}^{D_{emb} \times D_{emb}}$는 모든 헤드의 출력을 결합하여 최종 출력을 생성하는 선형 변환 행렬입니다.
FFN은 두 개의 선형 레이어와 GEGLU 활성화 함수로 구성된다. $$ \text{FFN}(x) = W_2(\text{GEGLU}(W_1(x))) $$


3.4. 분류기 (Classifier)
분류기 $\Phi_{cls}$는 최종 디코더 레이어의 출력 $Z = Q^{(L)} \in \mathbb{R}^{N_q \times D_{emb}}$를 입력받는다.
먼저 Projection4Classifier $\Phi_{proj}$를 통해 차원이 변환되고 Flatten된다. $$ \text{Features} = \text{Flatten}(\text{Projection}(Z)) \in \mathbb{R}^{N_q \times D_{feat}} $$

이후 간단한 MLP 분류기를 통과하여 최종 클래스 로짓을 출력한다. $$ \text{Logits} = \text{MLP}(\text{Features}) \in \mathbb{R}^{N_{cls}} $$



----시작



-----끝






4. 실험 (Experiments)

4.1. 데이터셋 (Datasets)
Sewer-ML: 사용한 공개 데이터셋의 특징, 클래스 구성('Normal', 'Defect') 설명.
데이터 분할: dataloader.py에 따라 훈련/검증/테스트 세트로 분할한 방식 설명.

4.2. 실험 환경 및 구현 세부사항
하이퍼파라미터 표: config.yaml의 주요 설정값을 표로 정리.
데이터 전처리: dataloader.py의 train_transform과 valid_test_transform에 적용된 데이터 증강 및 정규화 기법 상세 기술.
손실 함수 (Loss Function): 모델은 표준 교차 엔트로피(Cross-Entropy) 손실 함수 $\mathcal{L}{CE}$를 최소화하도록 훈련된다. $$ \mathcal{L}{CE} = -\sum_{i=1}^{N_{cls}} y_i \log(\hat{y}_i) $$ 여기서 $y_i$는 실제 레이블, $\hat{y}_i$는 모델의 예측 확률이다.

4.3. 평가 지표 (Evaluation Metrics)
분류 성능: Accuracy, Precision, Recall, F1-Score.
온디바이스 효율성: 4가지 핵심 지표 (파라미터 수, GFLOPs, 추론 시간, 메모리 사용량).

4.4. 실험 결과 및 분석
4.4.1. 베이스라인 모델과의 성능 비교
baseline.py에서 사용된 모델(ResNet-18, MobileNet-V2 등) 소개.

결과 표: 모든 모델에 대해 4.3에서 정의한 모든 평가 지표를 비교하는 종합 표 제시.
분석: 제안 모델이 훨씬 적은 파라미터와 연산량으로 베이스라인과 동등하거나 더 나은 F1-Score를 달성했음을 수치적으로 강조.

4.4.2. 어블레이션 스터디 (Ablation Study)
어텐션 풀링의 효과: attn_pooling을 false로 설정하고 고정 쿼리를 사용했을 때와 성능/파라미터 비교.

4.4.3. 기타 옵션 비교?
CNN 백본의 영향: CNN_feature_extractor를 다른 종류로 변경했을 때의 성능 변화 비교.
디코더 깊이의 영향: num_decoder_layers를 변경하며 성능과 효율성 트레이드오프 분석.

4.4.3. 정성적 분석: 어텐션 시각화 (Qualitative Analysis)
plot.py로 생성된 어텐션 맵 이미지 제시.
모델이 결함 이미지의 실제 결함 영역에 높은 어텐션 가중치를 부여하는 사례를 보여줌.




5. 결론 (Conclusion)
연구 요약: 제안한 LC-Transformer의 핵심 아이디어(경량 하이브리드 구조, 어텐션 풀링) 요약.
성과 강조: 실험을 통해 입증된 제안 모델의 높은 효율성과 경쟁력 있는 성능 강조.
시사점 및 기대효과: 온디바이스 AI 분야에 기여할 수 있는 실용적 가치 설명.
향후 연구 방향 (Future Work):
다중 결함 분류(Multi-label classification) 문제로의 확장.
결함 영역 분할(Segmentation) 태스크로의 발전 가능성 제시.
참고문헌 (References)
(인용한 모든 논문, 데이터셋, 소프트웨어 라이브러리 목록)
