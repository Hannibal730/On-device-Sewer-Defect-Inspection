import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from torch import Tensor
from torchvision import models

# =============================================================================
# 1. 이미지 인코더 모델 정의
# =============================================================================
class CnnFeatureExtractor(nn.Module):
    """
    다양한 CNN 아키텍처의 앞부분을 특징 추출기로 사용하는 범용 클래스입니다.
    config.yaml의 `cnn_feature_extractor.name` 설정에 따라 모델 구조가 결정됩니다.
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

        # --- MobileNetV4 (timm) ---
        elif cnn_feature_extractor_name == 'mobilenet_v4_feat1':
            base_model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True, out_indices=(0,))
            self._adjust_input_channels(base_model, in_channels)
            self.conv_front = base_model
            base_out_channels = 32 # feat1 출력 채널
        elif cnn_feature_extractor_name == 'mobilenet_v4_feat2':
            base_model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True, out_indices=(0, 1))
            self._adjust_input_channels(base_model, in_channels)
            self.conv_front = base_model
            base_out_channels = 48 # feat2 출력 채널
        elif cnn_feature_extractor_name == 'mobilenet_v4_feat3':
            base_model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True, out_indices=(0, 1, 2))
            self._adjust_input_channels(base_model, in_channels)
            self.conv_front = base_model
            base_out_channels = 64 # feat3 출력 채널
        elif cnn_feature_extractor_name == 'mobilenet_v4_feat4':
            base_model = timm.create_model('mobilenetv4_conv_small', pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
            self._adjust_input_channels(base_model, in_channels)
            self.conv_front = base_model
            base_out_channels = 96 # feat4 출력 채널

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
            elif 'mobilenetv4' in self.cnn_feature_extractor_name:
                # timm의 MobileNetV4 모델
                first_conv = base_model.conv_stem
                out_c, _, k, s, p, _, _, _ = first_conv.out_channels, first_conv.in_channels, first_conv.kernel_size, first_conv.stride, first_conv.padding, first_conv.dilation, first_conv.groups, first_conv.bias
                new_conv = nn.Conv2d(1, out_c, kernel_size=k, stride=s, padding=p, bias=False)
                with torch.no_grad():
                    new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                base_model.conv_stem = new_conv
        elif in_channels != 3:
            raise ValueError("in_channels는 1 또는 3만 지원합니다.")

    def forward(self, x):
        x = self.conv_front(x)
        x = self.conv_1x1(x) # 최종 채널 수 조정

        # timm의 features_only=True 모델은 리스트를 반환하므로 마지막 요소만 사용
        if isinstance(x, list):
            x = x[-1]

        return x

class PatchConvEncoder(nn.Module):
    """이미지를 패치로 나누고, 각 패치에서 특징을 추출하여 1D 시퀀스로 변환하는 인코더입니다."""
    def __init__(self, in_channels, img_size, patch_size, stride, featured_patch_dim, cnn_feature_extractor_name, pre_trained=True):
        super(PatchConvEncoder, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.featured_patch_dim = featured_patch_dim

        # stride를 고려한 패치 수 계산 (H, W 각각 계산)
        self.num_patches_H = (img_size - self.patch_size) // self.stride + 1
        self.num_patches_W = (img_size - self.patch_size) // self.stride + 1
        self.num_encoder_patches = self.num_patches_H * self.num_patches_W

        # 1. Shared CNN Feature Extractor
        self.shared_conv = nn.Sequential(
            CnnFeatureExtractor(cnn_feature_extractor_name=cnn_feature_extractor_name, pretrained=pre_trained, in_channels=in_channels, featured_patch_dim=featured_patch_dim),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1) # [B*num_encoder_patches, D]
        )
        
        # 2. Patch Mixer (Idea 2-1): 패치 간 정보 교환을 위한 Depthwise Convolution
        # 3x3 Depthwise Conv (groups=in_channels)는 파라미터가 매우 적습니다.
        # Padding=1을 주어 공간 크기(H_grid, W_grid)를 유지합니다.
        self.patch_mixer = nn.Sequential(
            nn.Conv2d(featured_patch_dim, featured_patch_dim, kernel_size=3, padding=1, groups=featured_patch_dim, bias=False),
            nn.BatchNorm2d(featured_patch_dim),
            nn.ReLU(inplace=True)
        )

        self.norm = nn.LayerNorm(featured_patch_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # 이미지를 패치로 분할
        patches = x.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, C, self.patch_size, self.patch_size)
        # patches: [B * num_patches, C, patch_size, patch_size]
        
        # 각 패치별 특징 추출
        conv_outs = self.shared_conv(patches) # [B * num_patches, D]
        
        # --- [Idea 2-1] Patch Mixing ---
        # 1. Grid 복원: [B * (H_p * W_p), D] -> [B, D, H_p, W_p]
        #    이 과정은 단순 View 연산으로 비용이 거의 들지 않습니다.
        #    하지만 이를 통해 인접 패치(상하좌우)가 누구인지 알 수 있게 됩니다.
        conv_outs_grid = conv_outs.view(B, self.num_patches_H, self.num_patches_W, self.featured_patch_dim).permute(0, 3, 1, 2)
        
        # 2. Mixing: Depthwise Conv로 인접 패치 정보 섞기
        mixed_outs = self.patch_mixer(conv_outs_grid)
        
        # 3. Flatten: 다시 시퀀스로 변환 [B, D, H_p, W_p] -> [B, H_p * W_p, D]
        #    permute(0, 2, 3, 1) -> [B, H_p, W_p, D]
        mixed_outs = mixed_outs.permute(0, 2, 3, 1).contiguous().view(B, -1, self.featured_patch_dim)

        # Layer Normalization 적용
        mixed_outs = self.norm(mixed_outs)
        
        return mixed_outs

# =============================================================================
# 2. 디코더 모델 정의
# =============================================================================

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class Embedding4Decoder(nn.Module): 
    """
    Decoder 입력을 위한 임베딩 레이어.
    [Idea 1-1] 2D Sinusoidal Position Encoding을 적용하여 위치 정보를 주입합니다.
    """
    def __init__(self, num_encoder_patches, featured_patch_dim, num_decoder_patches, 
                 grid_size_h, grid_size_w, # 2D PE 생성을 위해 그리드 크기 인자 추가
                 attn_pooling=False, num_decoder_layers=3, emb_dim=128, num_heads=16, 
                 decoder_ff_dim=256, attn_dropout=0., dropout=0., save_attention=False, res_attention=False, positional_encoding=True):
             
        super().__init__()
        
        self.attn_pooling = attn_pooling

        # --- 입력 인코딩 ---
        self.W_feat2emb = nn.Linear(featured_patch_dim, emb_dim)      
        self.dropout = nn.Dropout(dropout)

        # --- 학습 가능한 쿼리(Learnable Query) ---
        self.learnable_queries = nn.Parameter(torch.empty(num_decoder_patches, featured_patch_dim))
        nn.init.xavier_uniform_(self.learnable_queries)
        
        # --- [Idea 1-1] 2D Sinusoidal Positional Encoding ---
        self.use_positional_encoding = positional_encoding
        if self.use_positional_encoding:
            # 고정된 2D Sinusoidal PE 생성 (학습되지 않음)
            pos_embed = self.get_2d_sincos_pos_embed(emb_dim, grid_size_h, grid_size_w)
            # 모델의 state_dict에 저장되지만 optimizer에 의해 업데이트되지 않도록 register_buffer 사용
            self.register_buffer('pos_embed', pos_embed, persistent=False)
        else:
            self.pos_embed = None

        # --- 디코더 ---
        self.decoder = Decoder(num_encoder_patches, emb_dim, num_heads, num_decoder_patches, decoder_ff_dim=decoder_ff_dim, attn_dropout=attn_dropout, dropout=dropout,
                               res_attention=res_attention, num_decoder_layers=num_decoder_layers, save_attention=save_attention)
        
    def get_2d_sincos_pos_embed(self, embed_dim, grid_h, grid_w):
        """
        2D Grid에 대한 Sinusoidal Positional Embedding을 생성합니다.
        embed_dim의 절반은 Height 정보, 나머지 절반은 Width 정보에 할당합니다.
        """
        assert embed_dim % 2 == 0, "Embedding 차원은 짝수여야 합니다."
        
        # 1D Sinusoidal PE 생성 함수
        def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
            """
            embed_dim: output dimension for each position
            pos: [H*W] list of positions to be encoded
            """
            omega = torch.arange(embed_dim // 2, dtype=torch.float)
            omega /= embed_dim / 2.
            omega = 1. / 10000**omega  # (D/2,)

            pos = pos.reshape(-1)  # (M,)
            out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

            emb_sin = torch.sin(out) # (M, D/2)
            emb_cos = torch.cos(out) # (M, D/2)

            emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
            return emb

        # Grid 좌표 생성
        grid_h_arange = torch.arange(grid_h, dtype=torch.float)
        grid_w_arange = torch.arange(grid_w, dtype=torch.float)
        
        # Meshgrid로 좌표 확장
        grid_w_coords, grid_h_coords = torch.meshgrid(grid_w_arange, grid_h_arange, indexing='xy')
        
        # 각각에 대해 1D PE 생성 (차원의 절반씩 사용)
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h_coords) # [H*W, D/2]
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w_coords) # [H*W, D/2]

        # Concat하여 최종 2D PE 생성 [H*W, D]
        pos_embed = torch.cat([emb_h, emb_w], dim=1)
        return pos_embed.unsqueeze(0) # [1, H*W, D] (Broadcasting을 위해 배치 차원 추가)

    def forward(self, x) -> Tensor:
        # x: [B, num_encoder_patches, featured_patch_dim]
        bs = x.shape[0]

        x = self.W_feat2emb(x) # [B, N, emb_dim]

        # --- 위치 인코딩 더하기 ---
        if self.use_positional_encoding and self.pos_embed is not None:
            # self.pos_embed: [1, N, emb_dim] -> Broadcasting으로 더해짐
            x = x + self.pos_embed.to(x.device)

        seq_encoder_patches = self.dropout(x)
        
        # --- 2. 디코더에 입력할 쿼리(Query) 준비 ---
        if self.attn_pooling:
            latent_queries = self.W_feat2emb(self.learnable_queries)
            latent_queries = latent_queries.unsqueeze(0).repeat(bs, 1, 1)
            
            latent_attn_scores = torch.bmm(latent_queries, seq_encoder_patches.transpose(1, 2))
            latent_attn_weights = F.softmax(latent_attn_scores, dim=-1)
            
            seq_decoder_patches = torch.bmm(latent_attn_weights, seq_encoder_patches)
        else:
            learnable_queries = self.W_feat2emb(self.learnable_queries)
            seq_decoder_patches = learnable_queries.unsqueeze(0).repeat(bs, 1, 1)

        return seq_encoder_patches, seq_decoder_patches
            

class Projection4Classifier(nn.Module):
    """디코더의 출력을 받아 최종 분류기가 사용할 수 있는 특징 벡터로 변환합니다."""
    def __init__(self, emb_dim, featured_patch_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim, featured_patch_dim)
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x):
        x = self.linear(x)
        x = self.flatten(x)
        return x 
            
class Decoder(nn.Module):
    def __init__(self, num_encoder_patches, emb_dim, num_heads, num_decoder_patches, decoder_ff_dim=None, attn_dropout=0., dropout=0.,
                 res_attention=False, num_decoder_layers=1, save_attention=False):
        super().__init__()
        
        self.layers = nn.ModuleList([DecoderLayer(num_encoder_patches, emb_dim, num_decoder_patches, num_heads=num_heads, decoder_ff_dim=decoder_ff_dim, attn_dropout=attn_dropout, dropout=dropout,
                                                      res_attention=res_attention, save_attention=save_attention) for i in range(num_decoder_layers)])
        self.res_attention = res_attention

    def forward(self, seq_encoder:Tensor, seq_decoder:Tensor):
        scores = None
        if self.res_attention:
            for mod in self.layers: _, seq_decoder, scores = mod(seq_encoder, seq_decoder, prev=scores)
            return seq_decoder
        else:
            for mod in self.layers: _, seq_decoder = mod(seq_encoder, seq_decoder)
            return seq_decoder

class DecoderLayer(nn.Module):
    def __init__(self, num_encoder_patches, emb_dim, num_decoder_patches, num_heads, decoder_ff_dim=256, save_attention=False,
                 attn_dropout=0, dropout=0., bias=True, res_attention=False):
        super().__init__()
        assert not emb_dim%num_heads, f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})"
        
        self.res_attention = res_attention
        self.cross_attn = _MultiheadAttention(emb_dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention, qkv_bias=True)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(emb_dim)

        self.ffn = nn.Sequential(nn.Linear(emb_dim, decoder_ff_dim, bias=bias),
                                GEGLU(),
                                nn.Dropout(dropout),
                                nn.Linear(decoder_ff_dim//2, emb_dim, bias=bias)) 
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(emb_dim)
        
        self.save_attention = save_attention

    def forward(self, seq_encoder:Tensor, seq_decoder:Tensor, prev=None) -> Tensor:
        if self.res_attention:
            decoder_out, attn, scores = self.cross_attn(seq_decoder, seq_encoder, seq_encoder, prev)
        else:
            decoder_out, attn = self.cross_attn(seq_decoder, seq_encoder, seq_encoder)
        
        if self.save_attention:
            self.attn = attn
        
        seq_decoder = seq_decoder + self.dropout_attn(decoder_out)
        seq_decoder = self.norm_attn(seq_decoder)
        
        ffn_out = self.ffn(seq_decoder)
        seq_decoder = seq_decoder + self.dropout_ffn(ffn_out)  
        seq_decoder = self.norm_ffn(seq_decoder)
        
        if self.res_attention: return seq_encoder, seq_decoder, scores
        else: return seq_encoder, seq_decoder

class _MultiheadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, **kwargs):
        super().__init__()
        
        head_dim = emb_dim // num_heads
        self.scale = head_dim**-0.5
        self.num_heads, self.head_dim = num_heads, head_dim

        self.W_Q = nn.Linear(emb_dim, head_dim * num_heads, bias=qkv_bias)
        self.W_K = nn.Linear(emb_dim, head_dim * num_heads, bias=qkv_bias)
        self.W_V = nn.Linear(emb_dim, head_dim * num_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.concatheads2emb = nn.Sequential(nn.Linear(num_heads * head_dim, emb_dim), nn.Dropout(proj_dropout))

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, prev=None):
        bs = Q.size(0)
        
        q_s = self.W_Q(Q).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_s = self.W_K(K).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_s = self.W_V(V).view(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.einsum('bhqd, bhkd -> bhqk', q_s, k_s) * self.scale
        
        if prev is not None: attn_scores = attn_scores + prev
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.einsum('bhqk, bhkd -> bhqd', attn_weights, v_s)
        output = output.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.num_heads * self.head_dim)
        
        output = self.concatheads2emb(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class Model(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        
        num_encoder_patches = args.num_encoder_patches 
        num_labels = args.num_labels 
        num_decoder_patches = args.num_decoder_patches 
        self.featured_patch_dim = args.featured_patch_dim 
        attn_pooling = args.attn_pooling 
        emb_dim = args.emb_dim           
        num_heads = args.num_heads           
        num_decoder_layers = args.num_decoder_layers 
        decoder_ff_ratio = args.decoder_ff_ratio 
        dropout = args.dropout           
        attn_dropout = dropout           
        positional_encoding = args.positional_encoding 
        save_attention = args.save_attention     
        res_attention = getattr(args, 'res_attention', False)
        
        # 2D PE 생성을 위해 그리드 크기 전달 (PatchConvEncoder에서 계산된 값 필요)
        # 하지만 Model init 시점에는 Encoder가 이미 생성되어 있을 것이므로,
        # 아래와 같이 img_size, patch_size 등으로 다시 계산하거나, args에 담아 전달해야 함.
        # 여기서는 args에 grid 정보를 추가하는 방식이 깔끔하지만, 
        # run.py의 구조를 변경하지 않기 위해 직접 계산합니다.
        
        # models.py 내부에서 계산 (run.py 수정 최소화)
        # Model 클래스 호출 전에 args에 grid_size_h, w가 없으므로 계산 로직 추가 필요
        # 하지만 Encoder 객체는 run.py에서 생성되어 주입되는 것이 아니라, 
        # run.py에서 Model을 생성할 때 encoder는 별도로 생성됨. 
        # HybridModel에서 결합됨.
        
        # *** 수정: Embedding4Decoder는 Model 클래스 안에서 생성됨. ***
        # 따라서 여기서 계산해서 넘겨줘야 함.
        # args는 SimpleNamespace이므로 직접 계산해서 추가
        
        # 2D Grid 크기 계산 (PatchConvEncoder와 동일한 로직)
        # 만약 run.py에서 model_cfg를 그대로 넘겨준다면 img_size 등이 있을 것임.
        # 하지만 현재 코드는 args로 개별 필드만 넘겨받는 구조일 수도 있음.
        # run.py를 보면 decoder_args를 새로 만들어 넘김.
        # 따라서 run.py의 decoder_params 딕셔너리에 grid_size를 추가하는 것이 정석이나,
        # 사용자가 "run.py"는 수정 요청을 안했으므로 여기서 역산해야 함.
        # num_encoder_patches는 제곱수라고 가정 (정사각형 이미지/패치)
        grid_size = int(math.sqrt(num_encoder_patches))
        grid_size_h = grid_size
        grid_size_w = grid_size
        # 만약 직사각형 이미지라면 이 추론은 틀릴 수 있지만, config에서 img_size 하나만 받으므로 정사각형 가정.

        decoder_ff_dim = emb_dim * decoder_ff_ratio 

        self.embedding4decoder = Embedding4Decoder(num_encoder_patches=num_encoder_patches, featured_patch_dim=self.featured_patch_dim, num_decoder_patches=num_decoder_patches, 
                                grid_size_h=grid_size_h, grid_size_w=grid_size_w, # 추가된 인자
                                attn_pooling=attn_pooling,
                                num_decoder_layers=num_decoder_layers, emb_dim=emb_dim, num_heads=num_heads, decoder_ff_dim=decoder_ff_dim, positional_encoding=positional_encoding,
                                attn_dropout=attn_dropout, dropout=dropout,
                                res_attention=res_attention, save_attention=save_attention)

        self.projection4classifier = Projection4Classifier(emb_dim, self.featured_patch_dim)

    def forward(self, x): 
        # x: [B, num_encoder_patches, featured_patch_dim]
        # (PatchConvEncoder의 출력이 여기로 들어옴)
        
        seq_encoder_patches, seq_decoder_patches = self.embedding4decoder(x)
        z = self.embedding4decoder.decoder(seq_encoder_patches, seq_decoder_patches)
        features = self.projection4classifier(z)
        return features

# =============================================================================
# 3. 전체 모델 구성
# =============================================================================
class Classifier(nn.Module):
    """디코더 백본의 출력을 받아 최종 클래스 로짓으로 매핑하는 분류기입니다."""
    def __init__(self, num_decoder_patches, featured_patch_dim, num_labels, dropout):
        super().__init__()
        input_dim = num_decoder_patches * featured_patch_dim 
        hidden_dim = (input_dim + num_labels) // 2 

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x):
        x = self.projection(x) 
        return x

class HybridModel(torch.nn.Module):
    """인코더, 디코더, 분류기를 결합한 최종 하이브리드 모델입니다."""
    def __init__(self, encoder, decoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        
    def forward(self, x):
        # 1. 인코딩: 2D 이미지 -> 패치 시퀀스 (여기서 Mixer가 동작)
        x = self.encoder(x)
        # 2. 크로스-어텐션: 패치 시퀀스 -> 특징 벡터
        x = self.decoder(x)
        # 3. 분류: 특징 벡터 -> 클래스 로짓
        out = self.classifier(x)
        return out