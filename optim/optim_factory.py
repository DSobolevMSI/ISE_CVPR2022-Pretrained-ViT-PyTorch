import torch
import logging
from timm.optim import create_optimizer_v2

_logger = logging.getLogger(__name__)

def build_optimizer_for_aux_model(model, args):
    """
    通用优化器构建函数：支持 DeiTAux 和 SwinTransformerAux。
    为 'Scratch' (新增) 分支设定独立的学习率 (args.canny_lr)。
    
    Args:
        model: DeiTAux 或 SwinTransformerAux 模型
        args: 参数对象，需包含:
              - args.lr: 主干学习率 (Backbone)
              - args.canny_lr: 辅助分支学习率 (Scratch)
              - args.weight_decay: 全局权重衰减
              - args.opt: 优化器名称
    """
    
    # 1. 定义关键词：凡是参数名里包含这些词的，都算作"从头训练层"，使用 canny_lr
    # 兼容 DeiT (edge_embed) 和 Swin (canny_encoder)
    scratch_keywords = [
        'canny_encoder', # SwinAux / DeiTDual 用
        'edge_embed',    # DeiTAux 用
        'aux_head',      # 辅助头
        'fusion_head',   # 融合头
        'edge_head'      # 备用
    ]
    
    # 2. 获取模型自带的 no_weight_decay 列表 (如 pos_embed, cls_token, bias 等)
    skip = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
        
    backbone_params = []
    scratch_params = []
    
    # 3. 遍历参数进行分组
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 

        # --- 处理 Weight Decay ---
        # 按照 timm 惯例：1D 参数 (bias, norm)、pos_embed 等不进行 weight decay
        if param.ndim <= 1 or name.endswith(".bias") or name in skip:
            this_wd = 0.
        else:
            this_wd = args.weight_decay

        # 构建参数字典
        param_group = {
            'params': [param], 
            'weight_decay': this_wd, 
            'name': name
        }

        # --- 核心逻辑：区分 Backbone 和 Scratch ---
        if any(keyword in name for keyword in scratch_keywords):
            # >>> 命中关键词 -> 归为 Scratch 组，使用 canny_lr
            param_group['lr'] = args.canny_lr
            scratch_params.append(param_group)
        else:
            # >>> 未命中 -> 归为 Backbone 组，使用 args.lr
            param_group['lr'] = args.lr
            backbone_params.append(param_group)

    # 4. 打印分组统计信息 (Debug用)
    print(f"Optimizer Groups Summary:")
    print(f"  - [Backbone] (lr={args.lr:.1e}): {len(backbone_params)} tensors")
    print(f"  - [Scratch ] (lr={args.canny_lr:.1e}): {len(scratch_params)} tensors")
    
    # 验证是否遗漏
    if len(scratch_params) == 0:
        print("WARNING: No scratch parameters found! Check your layer names or keywords.")

    # 5. 合并列表并创建优化器
    all_params = backbone_params + scratch_params
    
    optimizer = create_optimizer_v2(
        all_params, 
        opt=args.opt, 
        lr=args.lr,  # 这里作为默认值传入，实际生效的是 param_group 里的 lr
        weight_decay=args.weight_decay, 
        momentum=getattr(args, 'momentum', 0.9)
    )
    
    return optimizer