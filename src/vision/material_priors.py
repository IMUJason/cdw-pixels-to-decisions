"""CDW 物料物理先验表（密度 t/m³、平均厚度 m）——伪真值管线用。

数值来源：工程手册/文献典型区间（混凝土/砖/木材密度为共识值；泡沫/塑料厚度
差异大故取宽区间）。作为**可配置先验**，发表前需补正式引用并做敏感性分析
（正是 E5 实验内容）。区间即显式建模的不确定性来源。

areal_density（面密度 t/m²）= density × thickness。
"""
from dataclasses import dataclass


@dataclass
class MaterialPrior:
    density_lo: float   # t/m³
    density_hi: float
    thick_lo: float     # m（传送带上碎料的平均可见厚度先验）
    thick_hi: float

    def sample(self, rng):
        d = rng.uniform(self.density_lo, self.density_hi)
        t = rng.uniform(self.thick_lo, self.thick_hi)
        return d * t  # 面密度 t/m²


# CODD 10 类先验（键与 codd_to_coco.CATS 一致）
CODD_PRIORS = {
    "concrete":     MaterialPrior(2.30, 2.40, 0.10, 0.20),
    "brick":        MaterialPrior(1.80, 2.00, 0.08, 0.15),
    "tile":         MaterialPrior(2.00, 2.20, 0.02, 0.05),
    "wood":         MaterialPrior(0.50, 0.70, 0.05, 0.15),
    "gypsum_board": MaterialPrior(0.70, 0.90, 0.03, 0.06),
    "foam":         MaterialPrior(0.03, 0.08, 0.05, 0.20),
    "general_w":    MaterialPrior(0.30, 0.60, 0.08, 0.20),
    "stone":        MaterialPrior(2.50, 2.70, 0.10, 0.25),
    "plastic":      MaterialPrior(0.90, 1.10, 0.02, 0.08),
    "pipes":        MaterialPrior(1.30, 1.45, 0.05, 0.12),
}

# 像素->米的几何标定：CODD 传送带相机视场宽约 1.0 m 对应 1920 px
# （可调参数；E5 敏感性分析对象之一）
CODD_METERS_PER_PIXEL = 1.0 / 1920.0

# CDW-Seg（Sirimewan 2025）类缩写 -> 全名（据 Scientific Data 论文十类匹配）
CDW_SEG_CLASS_NAMES = {
    "BIN": "skip_bin", "CB": "cardboard", "FD": "fill_dirt", "WT": "timber",
    "HP": "hard_plastic", "SP": "soft_plastic", "ST": "steel", "FB": "fabric",
    "CP": "concrete", "PB": "plasterboard",
}

# CDW-Seg 物料先验（BIN 是容器不是物料，组成估计中排除）
CDW_SEG_PRIORS = {
    "CP": MaterialPrior(2.30, 2.40, 0.10, 0.20),   # concrete
    "FD": MaterialPrior(1.40, 1.70, 0.10, 0.30),   # fill dirt（松散渣土）
    "WT": MaterialPrior(0.50, 0.70, 0.05, 0.15),   # timber
    "HP": MaterialPrior(0.90, 1.10, 0.02, 0.06),   # hard plastic
    "SP": MaterialPrior(0.03, 0.10, 0.02, 0.08),   # soft plastic（膜/袋）
    "ST": MaterialPrior(7.80, 8.00, 0.01, 0.05),   # steel
    "FB": MaterialPrior(0.15, 0.35, 0.02, 0.08),   # fabric
    "CB": MaterialPrior(0.55, 0.75, 0.02, 0.08),   # cardboard
    "PB": MaterialPrior(0.70, 0.90, 0.03, 0.06),   # plasterboard
}
# CDW-Seg 像素标定：3000x4000 相机拍整个 skip bin，视场约 3 m 宽对应 3000 px
CDW_SEG_METERS_PER_PIXEL = 3.0 / 3000.0
