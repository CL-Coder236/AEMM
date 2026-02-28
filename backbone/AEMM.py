# Copyright 2022-present, ...
import torch
import torch.nn as nn

import torchvision.transforms as transforms
import copy
from transformers import ViTModel
from backbone import MammothBackbone, register_backbone

class DARFM(nn.Module):
    def __init__(self, z_dim, attn_dim, num_heads = 8):
        super().__init__()

        self.proj = nn.Linear(z_dim, attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Linear(attn_dim, 2)

    def forward(self, z_concat):
        z = self.proj(z_concat)                       
        attn_out, _ = self.attn(z, z, z)            
        pooled = attn_out.mean(dim=1)                
        w = self.gate(pooled)                     
        return w

class ExpertModule(torch.nn.Module):
    def __init__(self, dim_invariant, dim_evolved, num_class=10, hidden_dim=1024):
        super().__init__()
        self.weight_fc = DARFM(z_dim=dim_invariant, attn_dim=512)

        self.fusion_fc = torch.nn.Sequential(
            torch.nn.Linear(dim_invariant + dim_evolved, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.LayerNorm(hidden_dim)
        )

        self.classifier = torch.nn.Linear(hidden_dim, num_class)

    def forward(self, invariant_token, evolved_token):
        invariant_cls_token = invariant_token[:, 0]   # [B, token_dim]
        evolved_cls_token = evolved_token[:, 0]       # [B, token_dim]


        combined = torch.cat([invariant_token, evolved_token], dim=1)  
        origin_weights = self.weight_fc(combined)     
        weights = torch.softmax(origin_weights, dim=1) 

        fused_feat = torch.cat([weights[:, 0:1] * invariant_cls_token,
                                weights[:, 1:2] * evolved_cls_token], dim=1)  

        x = self.fusion_fc(fused_feat)
        return self.classifier(x)

class DualBranchViT(nn.Module):
    def __init__(self, pretrained_vit, unfrozen_layers=3):
        super().__init__()
        self.unfrozen_layers = unfrozen_layers
        num_layers = len(pretrained_vit.encoder.layer)
        self.split_layer = num_layers - unfrozen_layers

        self.embeddings = pretrained_vit.embeddings

        self.shared_layers = nn.ModuleList(
            [pretrained_vit.encoder.layer[i] for i in range(0, self.split_layer)]
        )

        self.invariant_layers = nn.ModuleList(
            [pretrained_vit.encoder.layer[i] for i in range(self.split_layer, num_layers)]
        )

        self.evolved_layers = nn.ModuleList(
            [copy.deepcopy(layer) for layer in self.invariant_layers]
        )
        
    def forward(self, x):
        x = self.embeddings(pixel_values=x)
        assert isinstance(x, torch.Tensor), f"Expected tensor, got {type(x)}"
        for blk in self.shared_layers:
            x = blk(x)

        x_inv = x
        x_evo = x
        evolved_feats = []
        # ===== invariant branch =====
        with torch.no_grad():
            for blk in self.invariant_layers:
                x_inv = blk(x_inv)

        # ===== evolved branch =====
        for blk in self.evolved_layers:
            x_evo = blk(x_evo)
            evolved_feats.append(x_evo[:, 0])

        cls_inv = x_inv
        cls_evo = x_evo

        return cls_inv, cls_evo, evolved_feats

class AEMM(MammothBackbone):

    def __init__(self, num_classes: int) -> None:
        super(AEMM, self).__init__()
        self.device = "cpu"
        self.num_classes = num_classes
        self.unfrozen_layers = 3
        
        self.vitProcess = transforms.Compose(
            [transforms.Resize(224)])
        model = ViTModel.from_pretrained("google/vit-base-patch16-224", cache_dir = './weights', local_files_only=True)
        self.vitmodel = DualBranchViT(model, unfrozen_layers=self.unfrozen_layers)

        self.classifierArr = nn.ModuleList()
        self.currentTaskIndex = 0

    def createNewExpert(self, num_class = 200):
        print(f"Creating new expert for task {self.currentTaskIndex + 1}, with {num_class} classes.")
        newExpert = ExpertModule(dim_invariant = 768, dim_evolved = 768, num_class=num_class).to(self.device)
        self.classifierArr.append(newExpert)
        self.add_module(f"expert_{len(self.classifierArr) - 1}", newExpert)

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)
    
    def forward(self, x: torch.Tensor, returnt='out'):
        processX = self.vitProcess(x)
        # print("processX.shape======================",processX.shape)
        outputs_invariant, outputs_evolved, evolved_feats = self.vitmodel(processX)
        invariant_vit_feature = outputs_invariant
        evolved_vit_feature = outputs_evolved

        if returnt == 'feature':
            return evolved_feats
        
        out = self.classifierArr[self.currentTaskIndex](
            outputs_invariant, 
            outputs_evolved
        )
        if returnt == 'out':
            return out
        elif returnt == 'both':
            return out, evolved_feats
        elif returnt == 'features':
            return invariant_vit_feature, evolved_vit_feature
        
        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both'] but got {}".format(returnt))


@register_backbone("aemm")
def aemm_backbone(num_classes):
    return AEMM(num_classes)