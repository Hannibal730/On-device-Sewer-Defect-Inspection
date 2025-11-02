import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models

# =============================================================================
# 1. 이미지 인코더 모델 정의
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

# =============================================================================
# 2. CATS 디코더 모델 정의
# =============================================================================

# GEGLU (Gated Enhanced Gated Linear Unit) 액티베이션 함수를 구현한 클래스입니다.
# 일반적인 ReLU나 GELU와 달리, 입력의 일부를 게이트로 사용하여 동적으로 출력을 조절하는 특징이 있습니다.
class GEGLU(nn.Module):
    # 이 클래스의 순전파 로직을 정의합니다. 입력 텐서를 받아 GEGLU 연산을 수행합니다.
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        # 입력 텐서 `x`의 마지막 차원(`dim=-1`)을 기준으로 x를 두 개의 동일한 크기의 청크(chunk)로 나눕니다.
        # `x`는 주 데이터 경로가 되고, `gate`는 게이트 역할을 합니다. 수학적으로 x_in = [x, gate] 입니다.
        return x * F.gelu(gate)
        # 주 데이터 경로 `x`와, `gate`에 GELU(Gaussian Error Linear Unit) 활성화 함수를 적용한 결과를 요소별로 곱합니다.
        # 이 게이팅 메커니즘은 gate가 x에서 어떤 요소를 증폭하거나 줄일지 스스로 판단합니다.
        # 덕분에 모델이 더 복잡한 패턴을 학습하도록 돕습니다. 수학식은 Output = x * GELU(gate) 입니다.

# Query-Adaptive Masking (QAM)을 구현한 클래스입니다.
# 학습 중에 입력 텐서의 일부를 동적으로 마스킹(0으로 만듦)하여 레귤러라이제이션(regularization) 효과를 줍니다.
# 특히, 각 위치의 마스킹 확률을 매번 [start_prob, end_prob] 범위 내에서 무작위로 샘플링하여 적용합니다.
class QueryAdaptiveMasking(nn.Module):
    # QAM 클래스의 생성자입니다. 마스킹을 적용할 차원과 확률 범위를 설정합니다.
    def __init__(self, dim=1, start_prob =0.1, end_prob =0.5):
        super().__init__()
        # `nn.Module`의 생성자를 호출하여 PyTorch 모델로서의 기본 기능을 초기화합니다.
        self.dim = dim
        # 마스킹 목표 차원을 저장합니다. `dim=1`은 텐서의 두 번째 차원 N_T를 의미합니다.
        self.start_prob = start_prob
        # 마스킹 확률의 시작 값을 저장합니다. 이 값은 지정된 차원의 첫 번째 요소에 적용됩니다.
        self.end_prob = end_prob
        # 마스킹 확률의 끝 값을 저장합니다. 이 값은 지정된 차원의 마지막 요소에 적용됩니다.
    # QAM의 순전파 로직을 정의합니다.
    def forward(self, x):
        if not self.training:
            # 모델이 평가 모드(`model.eval()`)일 때는 마스킹을 적용하지 않습니다.
            return x
            # 입력을 그대로 반환하여 예측 시에는 일관된 결과를 얻도록 합니다.
        # 모델이 학습 모드(`model.train()`)일 때만 마스킹을 적용합니다.
        else:
            size = x.shape[self.dim]
            # 마스킹을 적용할 차원의 크기(예: 디코더 쿼리 패치 수)를 가져옵니다.

            # 각 위치에 적용될 마스킹 확률을 [start_prob, end_prob] 범위 내에서 무작위로 샘플링합니다.
            rand_probs = torch.rand(size, device=x.device) # [0, 1) 범위의 랜덤 값 생성
            dropout_prob = self.start_prob + (self.end_prob - self.start_prob) * rand_probs # 선형 보간
            
            # `.view`를 통해 확률 텐서의 모양을 입력 텐서 `x`와 브로드캐스팅이 가능하도록 조정합니다.
            dropout_prob = dropout_prob.view([-1 if i == self.dim else 1 for i in range(x.dim())])

            mask = torch.bernoulli(1 - dropout_prob).expand_as(x)
            # `1 - dropout_prob` 확률(p)에 따라 1(성공) 또는 0(실패)의 값을 갖는 베르누이 분포로부터 마스크를 생성합니다. 즉, 각 요소는 1-p의 확률로 1이 되고 p의 확률로 0이 됩니다.
            # 이 마스크는 입력 `x`와 동일한 크기로 확장됩니다.
            return x*mask
            # 생성된 마스크를 입력 텐서 `x`에 요소별로 곱하여 특정 요소들을 0으로 만듭니다(마스킹). 수학식은 x_out = x * mask 입니다.

# 이미지 분류 모델의 디코더 백본(backbone)을 정의하는 클래스입니다.
# 입력 패치와 학습 가능한 쿼리(learnable queries)를 임베딩하고 트랜스포머 디코더를 통해 예측을 수행하는 클래스입니다.
# 디코더에 입력할 seq_encoder_patches와 seq_decoder_patches를 생성합니다. 이후 디코더에 입력되어 특징 벡터를 생성하고, 오차 계산 및 역전파를 통해 훈련됩니다.
# 이 파라미터는 처음에는 무작위 값으로 시작하지만, 훈련 과정을 통해 분류에 중요한 특징을 추출하기 위한 유의미한 질문(쿼리)으로 학습되기 때문에 "학습 가능한 쿼리"라고 부릅니다.
class Embedding4Decoder(nn.Module): 
    # 클래스의 생성자입니다.
    def __init__(self, num_encoder_patches, featured_patch_dim, num_decoder_patches, attn_pooling=False, num_decoder_layers=3, emb_dim=128, num_heads=16, qam_prob_start=0.1, qam_prob_end=0.5,
                 decoder_ff_dim=256, attn_dropout=0., dropout=0., save_attention=False, res_attention=False, positional_encoding=True, **kwargs):
             
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        self.attn_pooling = attn_pooling

        # --- 입력 인코딩 ---
        self.W_feat2emb = nn.Linear(featured_patch_dim, emb_dim)      
        # 입력 패치(`featured_patch_dim` 차원)를 모델의 은닉 상태 차원(`emb_dim`)으로 변환하는 선형 레이어(가중치 `W_feat2emb`)를 정의합니다.
        self.dropout = nn.Dropout(dropout)
        # 일반적인 드롭아웃 레이어를 정의합니다.

        # --- 학습 가능한 쿼리(Learnable Query) 설정 (Xavier 초기화 적용) ---
        self.learnable_query = nn.Parameter(torch.empty(num_decoder_patches, featured_patch_dim))
        # Xavier 초기화는 훈련 초기 안정성을 높이고 수렴을 돕는 검증된 방법입니다.
        nn.init.xavier_uniform_(self.learnable_query)
        
        # --- 학습 가능한 위치 인코딩 ---
        # 입력 시퀀스의 위치 정보를 제공하기 위해, '학습 가능한 위치 인코딩(Positional Encoding)'을 파라미터로 생성합니다.
        self.use_positional_encoding = positional_encoding
        if self.use_positional_encoding:
            # 사인/코사인 함수 대신, 위치 정보를 담는 벡터 자체를 학습 파라미터로 사용합니다.
            # [num_encoder_patches, emb_dim] 크기의 텐서를 생성하며, 각 행은 특정 위치(패치)에 대한 위치 인코딩 벡터를 나타냅니다.
            # 이 파라미터는 훈련 과정에서 역전파를 통해 최적화됩니다.
            self.PE = nn.Parameter(torch.zeros(num_encoder_patches, emb_dim))
            nn.init.uniform_(self.PE, -0.02, 0.02) 
        else:
            self.PE = None
        # --- 디코더 ---
        self.decoder = Decoder(num_encoder_patches, emb_dim, num_heads, num_decoder_patches, decoder_ff_dim=decoder_ff_dim, attn_dropout=attn_dropout, dropout=dropout, 
                               qam_prob_start=qam_prob_start, qam_prob_end=qam_prob_end, res_attention=res_attention, num_decoder_layers=num_decoder_layers, save_attention=save_attention)
        
    # 순전파 로직을 정의합니다.
    def forward(self, x) -> Tensor:
        # 입력 x의 형태: [배치 크기, 인코더 패치 수, featured_patch_dim]
        bs = x.shape[0]

        # --- 1. 디코더에 입력할 입력 시퀀스 준비 (Key, Value) ---
        x = self.W_feat2emb(x)
        if self.use_positional_encoding:
            # 패치 특징 벡터에 학습 가능한 위치 인코딩(PE)을 더해줍니다.
            # x: [B, num_patches, emb_dim], PE: [num_patches, emb_dim] -> 브로드캐스팅을 통해 덧셈
            x = x + self.PE
        # x shape: [B, num_encoder_patches, emb_dim]

        seq_encoder_patches = self.dropout(x)
        # 인코딩된 입력 패치에 드롭아웃을 적용합니다.
        
        # --- 2. 디코더에 입력할 쿼리(Query) 준비 ---
        if self.attn_pooling:
            # [신규 방식] 어텐션 풀링을 이용한 파라미터-프리 동적 쿼리 생성
            # 1. 잠재 쿼리(latent query) 준비: learnable_query를 동적 쿼리 생성을 위한 씨앗(seed)으로 사용합니다.
            latent_query = self.W_feat2emb(self.learnable_query)
            latent_query = latent_query.unsqueeze(0).repeat(bs, 1, 1)
            
            # 2. 어텐션 스코어 계산 (Q=잠재 쿼리, K=패치 특징)
            latent_attn_scores = torch.bmm(latent_query, seq_encoder_patches.transpose(1, 2))
            latent_attn_weights = F.softmax(latent_attn_scores, dim=-1)
            
            # 3. 가중 평균으로 동적 쿼리 생성 (V=패치 특징)
            seq_decoder_patches = torch.bmm(latent_attn_weights, seq_encoder_patches)
        else:
            # [기존 방식] 고정된 학습 가능 쿼리 사용
            # 1. learnable_query를 임베딩
            learnable_query = self.W_feat2emb(self.learnable_query)
            # 2. 배치 크기만큼 복제하여 모든 샘플에 동일한 쿼리를 적용
            # learnable_query: [num_decoder_patches, emb_dim]
            # -> [1, num_decoder_patches, emb_dim]
            # -> [bs, num_decoder_patches, emb_dim]
            seq_decoder_patches = learnable_query.unsqueeze(0).repeat(bs, 1, 1)

        # Embedding 클래스는 이제 디코더에 필요한 입력 시퀀스들을 반환합니다.
        # 실제 디코더 호출은 Model 클래스의 forward에서 이루어집니다.
        return seq_encoder_patches, seq_decoder_patches
            

class Projection4Classifier(nn.Module):
    """디코더의 출력을 받아 최종 분류기가 사용할 수 있는 특징 벡터로 변환합니다."""
    def __init__(self, emb_dim, featured_patch_dim):
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        self.linear = nn.Linear(emb_dim, featured_patch_dim)
        # 트랜스포머의 은닉 상태 차원(`emb_dim`)을 `featured_patch_dim`으로 변환하는 선형 레이어를 정의합니다.
        self.flatten = nn.Flatten(start_dim=-2)
        # 디코더 패치들을 하나의 벡터로 펼치기 위한 Flatten 레이어를 정의합니다.

    def forward(self, x):
        # 입력 x의 형태: [B, num_decoder_patches, emb_dim]
        x = self.linear(x)
        # 입력 `x`를 선형 레이어에 통과시켜 차원을 변환합니다.
        # 결과: [B, num_decoder_patches, featured_patch_dim]
        
        # flatten을 적용하여 마지막 두 차원을 하나로 합칩니다.
        # [B, num_decoder_patches, D] -> [B, num_decoder_patches * D]
        x = self.flatten(x)
        return x # [B, num_decoder_patches * featured_patch_dim] 형태의 2D 텐서를 반환합니다.
            
# 여러 개의 디코더 레이어로 구성된 트랜스포머 디코더 클래스입니다.
class Decoder(nn.Module):
    # 디코더의 생성자입니다.
    def __init__(self, num_encoder_patches, emb_dim, num_heads, num_decoder_patches, decoder_ff_dim=None, attn_dropout=0., dropout=0., qam_prob_start = 0.1, qam_prob_end =0.5, 
                        res_attention=False, num_decoder_layers=1, save_attention=False):
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        self.layers = nn.ModuleList([DecoderLayer(num_encoder_patches, emb_dim, num_decoder_patches, num_heads=num_heads, decoder_ff_dim=decoder_ff_dim, qam_prob_start=qam_prob_start,
                                                      qam_prob_end=qam_prob_end, attn_dropout=attn_dropout, dropout=dropout,
                                                      res_attention=res_attention, save_attention=save_attention) for i in range(num_decoder_layers)])
        # `num_decoder_layers` 개수만큼의 `DecoderLayer`를 `nn.ModuleList`로 묶어 관리합니다.
        
        self.res_attention = res_attention
        # 잔차 어텐션(어텐션 스코어를 다음 레이어에 더해주는 기법) 메커니즘 사용 여부를 저장합니다.
    # 디코더의 순전파 로직을 정의합니다.
    def forward(self, seq_encoder:Tensor, seq_decoder:Tensor):
        scores = None
        # 잔차 어텐션에서 이전 레이어의 어텐션 스코어를 전달하기 위한 변수를 초기화합니다.
        
        if self.res_attention:
            # 잔차 어텐션을 사용하는 경우,
            for mod in self.layers: _, seq_decoder, scores = mod(seq_encoder, seq_decoder, prev=scores)
            return seq_decoder
        
        else:
            # 잔차 어텐션을 사용하지 않는 경우,
            for mod in self.layers: _, seq_decoder = mod(seq_encoder, seq_decoder)
            return seq_decoder

# 트랜스포머 디코더의 단일 레이어를 정의하는 클래스입니다.
# 크로스-어텐션(Cross-Attention)과 피드포워드 네트워크(Feed-Forward Network)로 구성됩니다.
class DecoderLayer(nn.Module):
    # 디코더 레이어의 생성자입니다.
    def __init__(self, num_encoder_patches, emb_dim, num_decoder_patches, num_heads, decoder_ff_dim=256, save_attention=False, qam_prob_start = 0.1, qam_prob_end =0.5, 
                 attn_dropout=0, dropout=0., bias=True, res_attention=False, **kwargs):
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        assert not emb_dim%num_heads, f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})"
        # `emb_dim`이 `num_heads`로 나누어 떨어져야 멀티헤드 어텐션이 가능하므로, 이를 확인합니다.
        
        # --- 크로스-어텐션 블록 ---
        self.res_attention = res_attention
        # 잔차 어텐션 사용 여부를 저장합니다.
        self.cross_attn = _MultiheadAttention(emb_dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention, qkv_bias=True)
        # 멀티헤드 크로스-어텐션 모듈을 초기화한다.
        self.dropout_attn = QueryAdaptiveMasking(dim=1, start_prob=qam_prob_start, end_prob=qam_prob_end)
        # 어텐션 출력에 적용할 Query-Adaptive Masking 레이어를 정의합니다.
        self.norm_attn = nn.LayerNorm(emb_dim)
        # 어텐션 블록의 잔차 연결(add) 후 적용될 레이어 정규화(Layer Normalization)를 정의합니다.

        # --- 피드포워드 네트워크 블록 ---
        # 위치별 피드포워드 네트워크(FFN)를 `nn.Sequential`로 정의합니다.
        self.ffn = nn.Sequential(nn.Linear(emb_dim, decoder_ff_dim, bias=bias), # 1. emb_dim -> decoder_ff_dim 확장
                                GEGLU(),                             # 2. GEGLU 활성화 함수 (이 활성함수를 거친 직후 차원이 절반이 됨))
                                nn.Dropout(dropout),                 # 3. 드롭아웃
                                nn.Linear(decoder_ff_dim//2, emb_dim, bias=bias)) # 4. decoder_ff_dim/2 -> emb_dim 축소 
        self.dropout_ffn = QueryAdaptiveMasking(dim=1, start_prob=qam_prob_start, end_prob=qam_prob_end)
        # FFN 출력에 적용할 Query-Adaptive Masking 레이어를 정의합니다.
        self.norm_ffn = nn.LayerNorm(emb_dim)
        # FFN 블록의 잔차 연결(add) 후 적용될 레이어 정규화를 정의합니다.
        
        self.save_attention = save_attention
        # 어텐션 가중치를 시각화 등의 목적으로 저장할지 여부를 결정합니다.

    # 디코더 레이어의 순전파 로직을 정의합니다.
    def forward(self, seq_encoder:Tensor, seq_decoder:Tensor, prev=None) -> Tensor:
        # `seq_decoder`는 쿼리(Q), `seq_encoder`는 키(K)와 값(V)으로 사용됩니다.
        
        # --- 멀티헤드 크로스-어텐션 ---
        if self.res_attention:
            # 잔차 어텐션을 사용하는 경우,
            decoder_out, attn, scores = self.cross_attn(seq_decoder, seq_encoder, seq_encoder, prev)
            # 어텐션 모듈은 (출력, 어텐션 가중치, 어텐션 스코어)를 반환합니다.
        else:
            # 잔차 어텐션을 사용하지 않는 경우,
            decoder_out, attn = self.cross_attn(seq_decoder, seq_encoder, seq_encoder)
            # 어텐션 모듈은 (출력, 어텐션 가중치)를 반환합니다.
        if self.save_attention:
            # 어텐션 가중치를 저장하도록 설정된 경우,
            self.attn = attn
            # 계산된 어텐션 가중치를 `self.attn`에 저장합니다.
        
        # --- 첫 번째 Add & Norm ---
        seq_decoder = seq_decoder + self.dropout_attn(decoder_out)
        # 1번째 Residual Connection: 어텐션 출력에 드롭아웃을 적용하고, 이를 입력에 더합니다.
        seq_decoder = self.norm_attn(seq_decoder)
        # 레이어 정규화를 적용합니다.
        
        # --- 피드포워드 네트워크 ---
        ffn_out = self.ffn(seq_decoder)
        # 정규화된 결과를 FFN에 통과시킵니다.

        # --- 두 번째 Add & Norm ---
        seq_decoder = seq_decoder + self.dropout_ffn(ffn_out)  
        # 2번째 Residual Connection: FFN의 출력에 드롭아웃을 적용하고, 이를 FFN의 입력에 더합니다.
        seq_decoder = self.norm_ffn(seq_decoder)
        # 레이어 정규화를 적용합니다.
        
        if self.res_attention: return seq_encoder, seq_decoder, scores
        else: return seq_encoder, seq_decoder

# 멀티헤드 어텐션 메커니즘을 구현한 내부 클래스입니다.
class _MultiheadAttention(nn.Module):
    # 멀티헤드 어텐션의 생성자입니다.
    def __init__(self, emb_dim, num_heads, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, **kwargs):
        """
        멀티헤드 어텐션 레이어
        입력 형태:
            Q (쿼리): [배치 크기, 디코더 패치 수, emb_dim]
            K (키), V (값): [배치 크기, 인코더 패치 수, emb_dim]
        """
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        head_dim = emb_dim // num_heads
        # 각 어텐션 헤드의 차원을 계산합니다.
        self.scale = head_dim**-0.5
        # 어텐션 스코어를 스케일링하기 위한 팩터입니다. d_k의 제곱근의 역수(1/sqrt(d_k))를 사용합니다.
        # 이는 Q, K 내적 값의 분산이 d_k에 비례하여 커지는 것을 방지하여, softmax 함수의 기울기 소실(gradient vanishing) 문제를 완화합니다.
        self.num_heads, self.head_dim = num_heads, head_dim
        # 헤드의 수와 각 헤드의 차원을 저장합니다.

        # 입력 Q, K, V를 `emb_dim` 차원에서 `num_heads * head_dim` 차원으로 변환하는 선형 레이어들을 정의합니다.
        self.W_Q = nn.Linear(emb_dim, head_dim * num_heads, bias=qkv_bias)
        self.W_K = nn.Linear(emb_dim, head_dim * num_heads, bias=qkv_bias)
        self.W_V = nn.Linear(emb_dim, head_dim * num_heads, bias=qkv_bias)

        self.res_attention = res_attention
        # 잔차 어텐션 사용 여부를 저장합니다.
        self.attn_dropout = nn.Dropout(attn_dropout)
        # 계산된 어텐션 가중치에 적용될 드롭아웃 레이어를 정의합니다.
        
        # 여러 헤드의 출력을 합친 벡터를 최종 임베딩 차원으로 변환하는 출력 레이어를 정의합니다.
        self.concatheads2emb = nn.Sequential(nn.Linear(num_heads * head_dim, emb_dim), nn.Dropout(proj_dropout))

    # 멀티헤드 어텐션의 순전파 로직을 정의합니다.
    def forward(self, Q:Tensor, K:Tensor, V:Tensor, prev=None):
        bs = Q.size(0)
        # 입력 쿼리(Q)에서 배치 크기를 가져옵니다.
        
        # --- Q, K, V 선형 변환 및 헤드 분할 --- 
        # 1. 선형 변환: [B, num_patches, emb_dim] -> [B, num_patches, num_heads * head_dim]
        # 2. view: [B, num_patches, num_heads, head_dim]
        # 3. permute: [B, num_heads, num_patches, head_dim] (einsum을 위한 차원 재배열)
        q_s = self.W_Q(Q).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_s = self.W_K(K).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_s = self.W_V(V).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # --- 어텐션 스코어 계산 ---
        attn_scores = torch.einsum('bhqd, bhkd -> bhqk', q_s, k_s) * self.scale
        # `torch.einsum` (아인슈타인 표기법)을 사용하여 Q와 K의 내적을 효율적으로 계산합니다.
        # 계산된 스코어를 `scale` 팩터로 스케일링합니다. Attention(Q, K) = (Q * K^T) / sqrt(d_k)
        
        if prev is not None: attn_scores = attn_scores + prev
        # 만약 이전 레이어의 어텐션 스코어(`prev`)가 주어지면, 현재 스코어에 더합니다 (잔차 어텐션).
        
        # --- 어텐션 가중치 및 출력 계산 ---
        attn_weights = F.softmax(attn_scores, dim=-1)
        # 어텐션 스코어에 소프트맥스 함수를 적용하여 확률 분포 형태의 어텐션 가중치를 얻습니다. 각 쿼리에 대한 키의 중요도를 나타냅니다.
        attn_weights = self.attn_dropout(attn_weights)
        # 어텐션 가중치에 드롭아웃을 적용합니다.

        output = torch.einsum('bhqk, bhkd -> bhqd', attn_weights, v_s)
        # 어텐션 가중치와 값(V)을 `einsum`으로 곱하여 최종 출력을 계산합니다.
        output = output.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.num_heads * self.head_dim)
        # 차원을 원래대로 복원 [B, num_patches, emb_dim] 하고, 헤드들을 다시 하나의 텐서로 합칩니다.
        
        output = self.concatheads2emb(output)
        # 최종 출력 레이어를 통과시킵니다.

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

# 전체 모델을 구성하고 순전파를 정의하는 메인 클래스입니다.
# 하이퍼파라미터를 인자로 받아 `Embedding`과 `Decoder2Classifier`를 직접 초기화합니다.
class Model(nn.Module):
    # 전체 모델의 생성자입니다.
    def __init__(self, args, **kwargs):
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        # --- 하이퍼파라미터 로드 ---
        # `args` 객체로부터 모델 구성에 필요한 모든 하이퍼파라미터를 가져옵니다.
        num_encoder_patches = args.num_encoder_patches # 인코더 패치의 수
        num_labels = args.num_labels # 예측할 클래스의 수
        num_decoder_patches = args.num_decoder_patches # 디코더 쿼리의 수 (YAML에서 설정)
        self.featured_patch_dim = args.featured_patch_dim # 각 패치의 특징 차원
        attn_pooling = args.attn_pooling # 어텐션 풀링 사용 여부
        emb_dim = args.emb_dim           # 모델의 은닉 상태 차원
        num_heads = args.num_heads           # 멀티헤드 어텐션의 헤드 수
        num_decoder_layers = args.num_decoder_layers # 트랜스포머 디코더의 레이어 수
        decoder_ff_ratio = args.decoder_ff_ratio # FFN 내부 차원 비율
        dropout = args.dropout           # 드롭아웃 비율
        attn_dropout = dropout           # 어텐션 드롭아웃도 동일한 비율 사용
        positional_encoding = args.positional_encoding # 위치 인코딩 사용 여부
        save_attention = args.save_attention     # 어텐션 가중치 저장 여부
        qam_prob_start = getattr(args, 'qam_prob_start', 0.0) # QAM 시작 확률
        qam_prob_end = getattr(args, 'qam_prob_end', 0.0)     # QAM 끝 확률
        res_attention = getattr(args, 'res_attention', False) # res_attention 사용 여부

        # FFN의 내부 차원을 계산합니다.
        decoder_ff_dim = emb_dim * decoder_ff_ratio # 예: 24 * 2 = 48

        # --- 백본 모델(임베딩 및 디코더) 초기화 --- 
        self.embedding4decoder = Embedding4Decoder(num_encoder_patches=num_encoder_patches, featured_patch_dim=self.featured_patch_dim, num_decoder_patches=num_decoder_patches, attn_pooling=attn_pooling,
                                num_decoder_layers=num_decoder_layers, emb_dim=emb_dim, num_heads=num_heads, decoder_ff_dim=decoder_ff_dim, positional_encoding=positional_encoding,
                                attn_dropout=attn_dropout, dropout=dropout, qam_prob_start=qam_prob_start, qam_prob_end=qam_prob_end, 
                                res_attention=res_attention, save_attention=save_attention, **kwargs)

        # 백본의 출력을 최종 분류기가 사용할 특징 벡터로 변환하는 헤드를 생성합니다.
        self.projection4classifier = Projection4Classifier(emb_dim, self.featured_patch_dim)

    # 전체 모델의 순전파 로직을 정의합니다.
    def forward(self, x): # 입력 x의 형태: [배치 크기, 인코더 패치 수, 특징 차원]

        # 1. Embedding: 인코더 패치와 학습 가능한 쿼리를 임베딩하여 디코더 입력 시퀀스 준비
        seq_encoder_patches, seq_decoder_patches = self.embedding4decoder(x)
        # 2. Decoder: 준비된 시퀀스들을 디코더에 통과시켜 쿼리 기반 특징 추출
        z = self.embedding4decoder.decoder(seq_encoder_patches, seq_decoder_patches)
        # 3. Projection4Classifier: 최종 특징 벡터로 변환
        features = self.projection4classifier(z)
        # 결과 features의 형태: [B, num_decoder_patches * featured_patch_dim]
        return features

# =============================================================================
# 3. 전체 모델 구성
# =============================================================================
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