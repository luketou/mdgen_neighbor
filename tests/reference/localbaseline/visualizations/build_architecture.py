#!/usr/bin/env python3
"""Build the offline MDGen architecture atlas from source-anchored graph specs.

Run with an environment containing matplotlib and plotly. No model import,
checkpoint loading, GPU, network access or training is needed.
"""

from __future__ import annotations

import hashlib
import heapq
import html
import io
import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from plotly.offline import get_plotlyjs

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
WIDTH, HEIGHT, NW, NH = 1640, 1000, 250, 110
LM = "mdgen/model/latent_model.py"
LAYERS = "mdgen/model/layers.py"
IPA = "mdgen/model/ipa.py"
HYENA = "mdgen/model/standalone_hyena.py"
LOCAL = "mdgen/full_length/attention.py"
SUMMARY = "mdgen/full_length/summaries.py"
PIPE = "mdgen/full_length/pipeline.py"
LEGEND = {
    "input": {"fill": "#d5e8d4", "stroke": "#427459", "label": "輸入／輸出"},
    "linear": {"fill": "#ffe6cc", "stroke": "#a76524", "label": "線性投影／MLP"},
    "spatial": {"fill": "#e1d5e7", "stroke": "#79588b", "label": "空間注意力"},
    "geometry": {"fill": "#dae8fc", "stroke": "#4c74a4", "label": "三維幾何／分區"},
    "temporal": {"fill": "#fff2cc", "stroke": "#a38330", "label": "時間混合"},
    "norm": {"fill": "#eef0f3", "stroke": "#747d8a", "label": "重排／正規化"},
    "add": {"fill": "#f8cecc", "stroke": "#a85553", "label": "相加／閘門"},
    "condition": {"fill": "#d5eeea", "stroke": "#387e76", "label": "條件／控制"},
}


def node(id, label, zh, shape, description, file, anchor, col, row,
         kind="linear", optional=False):
    lines = (ROOT / file).read_text().splitlines()
    matches = [i + 1 for i, line in enumerate(lines) if anchor in line]
    if not matches:
        raise ValueError(f"Missing source anchor: {file}: {anchor}")
    return dict(id=id, label=label, zh=zh, shape=shape, description=description,
                source=dict(file=file, line=matches[0], anchor=anchor),
                kind=kind, optional=optional, x=50 + col * 310, y=180 + row * 240)


def edge(source, target, label="", kind="data"):
    return dict(source=source, target=target, label=label, kind=kind)


def chart(id, title, subtitle, nodes, edges, notes):
    if id == "02-latent-layer":
        # Repeated symbol names also exist in IPALayer. Anchor this page to
        # LatentMDGenLayer itself rather than the earlier class definition.
        lines = (ROOT / LM).read_text().splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("class LatentMDGenLayer"))
        for n in nodes:
            if n["source"]["file"] == LM:
                anchor = n["source"]["anchor"]
                n["source"]["line"] = next(i + 1 for i in range(start, len(lines)) if anchor in lines[i])
    return dict(id=id, title=title, subtitle=subtitle, nodes=nodes, edges=edges, notes=notes)


def charts():
    return [
        chart("01-overview", "01 · MDGen 生成模型總覽",
              "從含雜訊的軌跡表示，到傳輸模型預測；由上方條件支路控制主幹。", [
            node("x", "Noisy latent x(t)", "含雜訊的軌跡", "[B,T,L,D]", "t 是 flow matching 的時間；T 是軌跡影格數。這兩個時間概念不同。", LM, "def forward(self, x, t, mask,", 0, 1, "input"),
            node("embed", "latent_to_emb", "將潛在值轉為特徵", "D → C; [B,T,L,C]", "Linear 將每個影格、每個殘基的 D 維值轉成 C 維特徵。可選絕對位置與時間編碼在此加入。", LM, "x = self.latent_to_emb(x)", 1, 1),
            node("condition", "cond_to_emb + mask_to_emb", "編碼已知條件與遮罩", "[B,T,L,C]", "x_cond 存在時，條件線性投影與二元遮罩嵌入逐元素加入軌跡特徵。", LM, "x = x + self.cond_to_emb", 1, 0, "condition", True),
            node("frames", "Frames / aatype", "起終點座標與胺基酸", "Rigid [B,L]; aatype [B,L]", "sim_condition 使用起始構形。TPS 使用起終點相對剛體變換，經雙支路 IPA 編碼後相加。", LM, "def run_ipa(", 0, 0, "geometry"),
            node("prepend", "IPALayer × N", "前置幾何條件編碼", "[B,L,C] → broadcast over T", "由 prepend_ipa 啟用。先編碼已知構形，得到 [B,L,C]；插入時間維後廣播給所有影格。詳見視圖 03。", LM, "if self.args.prepend_ipa:  # IPA", 2, 0, "geometry", True),
            node("fuse", "Feature addition", "加入條件與幾何編碼", "[B,T,L,C]", "圖中將前置條件加總合併為一個節點；程式先加入條件及遮罩，再加入 run_ipa 的結果。", LM, "x = x + self.run_ipa(", 2, 1, "add"),
            node("time", "TimestepEmbedder", "編碼生成進度 t", "[B] → [B,1,C]", "t 先乘 time_multiplier（預設 100），經 sin/cos → Linear → SiLU → Linear。它產生 AdaLN 控制值，不直接取代軌跡時間軸。", LM, "t = self.t_embedder", 3, 0, "condition"),
            node("stack", "LatentMDGenLayer × N", "重複空間、時間與 MLP", "[B,T,L,C] → [B,T,L,C]", "依序執行 N 個獨立參數的層。每層先可選 IPA，再空間注意力、時間混合、MLP。詳見視圖 02。", LM, "for layer_idx, layer in enumerate", 3, 1, "spatial"),
            node("head", "FinalLayer", "正規化、調變、輸出投影", "C → D; [B,T,L,D]", "LayerNorm（無 affine）→ 以 t 產生 shift/scale → Linear。此輸出頭沒有主幹那種 gate 殘差。", LAYERS, "class FinalLayer", 4, 1),
            node("output", "Transport prediction", "輸出速度場等訓練目標", "[B,T,L,D]", "預設 prediction=velocity。單次網路前向輸出傳輸目標，取樣器要反覆呼叫此網路後才得到完整生成軌跡。", LM, "latent = self.emb_to_latent", 4, 2, "input"),
            node("design", "Design head", "可選的胺基酸類別預測", "mean over T → [B,L,20]", "design 分支：fc1 → GELU → fc2 → 時間平均 → fc3 → GELU → 20 類 logits；一般 design 加到 latent 最後 20 維。MPNN 分支直接回傳 logits。", LM, "x_l = self.fc2(gelu(self.fc1(x)))", 3, 2, "linear", True),
        ], [edge("x", "embed"), edge("embed", "fuse"), edge("condition", "fuse", "條件相加", "condition"),
            edge("frames", "prepend", "已知構形", "condition"), edge("prepend", "fuse", "沿 T 廣播", "optional"),
            edge("fuse", "stack"), edge("time", "stack", "AdaLN", "condition"), edge("time", "head", "shift / scale", "condition"),
            edge("time", "prepend", "AdaLN", "condition"),
            edge("stack", "head"), edge("head", "output"), edge("stack", "design", "design", "optional"),
            edge("design", "output", "末 20 維", "optional")], [
                "符號：B 批次、T 軌跡影格、L 殘基、C 隱藏特徵、D 潛在維度、N 主幹層數。",
                "解析器預設：C=384、N=5、T=50；一般 D=21，TPS 等為 28，design 加 20，no_frames 為 111。",
                "IPA／Hyena／local_spatial 皆由選項啟用；本圖涵蓋分支，不代表所有選項預設同時開啟。",
            ]),
        chart("02-latent-layer", "02 · 展開一個 LatentMDGenLayer",
              "依箭頭走：上排向右，中排向左，下排向右。三次閘門殘差保留輸入。", [
            node("x", "Layer input", "第 k 層的輸入特徵", "[B,T,L,C]", "每層維持相同的主幹張量形狀。", LM, "B, T, L, C = x.shape", 0, 0, "input"),
            node("ipa", "LayerNorm → IPA + x", "可選的幾何更新", "[B,T,L,C]", "interleave_ipa 啟用後，在每個主幹層最前面加入 IPA(norm(x), start_frames)。這個殘差沒有時間 gate。", LM, "if hasattr(self, 'ipa')", 1, 0, "geometry", True),
            node("snorm", "LayerNorm + AdaLN-L", "空間子層的特徵調變", "(1 + scale_L) · LN(x) + shift_L", "沿特徵 C 正規化；生成時間 t 提供空間專用 shift 和 scale。", LM, "x = modulate(self.mha_layer_norm(x), shift_msa_l", 2, 0, "norm"),
            node("spatial", "Spatial attention", "同一影格的殘基互看", "[B·T,L,C] → [B,T,L,C]", "dense 使用 AttentionWithRoPE，關聯 L 個殘基；local_spatial 使用 LocalSelfAttention 的 core/ghost 分組。", LM, "x.reshape(B * T, L, C)", 3, 0, "spatial"),
            node("sadd", "Residual + gate-L", "保留原值並加入空間更新", "x + gate_L · spatial", "gate 由 AdaLN_modulation(t) 產生。圖上虛線為跳接，不是另一個注意力層。", LM, "x = residual + gate_msa_l.unsqueeze(1) * x", 4, 0, "add"),
            node("summary", "Cell summary exchange", "可選的跨區域資訊交流", "[B,T,L,C]", "local_spatial 時執行；每區 core 平均 → 摘要間全域注意力 → 加回 owner 殘基。詳見視圖 06。", LM, "self.cell_summaries(x[batch_index], partition)", 4, 1, "spatial", True),
            node("tnorm", "LayerNorm + AdaLN-T", "時間子層的特徵調變", "(1 + scale_T) · LN(x) + shift_T", "程式重用 mha_layer_norm 模組，但 shift/scale/gate 使用獨立的時間組。", LM, "x = modulate(self.mha_layer_norm(x), shift_msa_t", 3, 1, "norm"),
            node("temporal", "Time attention / Hyena", "同一殘基的影格互相交流", "[B·L,T,C] → [B,T,L,C]", "先轉置 T/L，合併 B 與 L。hyena=False 使用 AttentionWithRoPE；True 使用 HyenaOperator。兩種方法互斥，Hyena 要求無 padding。", LM, "x.transpose(1, 2).reshape(B * L, T, C)", 2, 1, "temporal"),
            node("tadd", "Residual + gate-T", "保留原值並加入時間更新", "x + gate_T · temporal", "混合後還原成 [B,T,L,C]，乘時間 gate 再殘差相加。", LM, "x = residual + gate_msa_t.unsqueeze(1) * x", 1, 1, "add"),
            node("fnorm", "LayerNorm + AdaLN-MLP", "前饋網路的特徵調變", "[B,T,L,C]", "MLP 的正規化與 shift/scale 獨立於空間／時間的調變參數。", LM, "x = modulate(self.final_layer_norm(x), shift_mlp", 0, 1, "norm"),
            node("fc1", "fc1 → GELU", "擴展每個 token 的通道", "C → 4C", "逐位置線性層加 GELU。預設 384 → 1536；不混合時間或殘基位置。", LM, "self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)", 0, 2),
            node("fc2", "fc2", "將通道縮回原始維度", "4C → C", "前饋網路第二個 Linear；預設 1536 → 384。", LM, "self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)", 1, 2),
            node("fadd", "Residual + gate-MLP", "輸出給下一個主幹層", "[B,T,L,C]", "x + gate_MLP × MLP。完成此層後進入下一個 LatentMDGenLayer，最後才進 FinalLayer。", LM, "x = residual + gate_mlp.unsqueeze(1) * x", 2, 2, "add"),
            node("time", "SiLU → Linear(C, 9C)", "由 t 產生三組調變參數", "3 × (shift, scale, gate)", "空間、時間、MLP 各取得自己的 shift/scale/gate。主幹 AdaLN 的最後線性層以零初始化。", LM, "nn.Linear(self.embed_dim, 9 * self.embed_dim", 4, 2, "condition"),
        ], [edge("x", "ipa"), edge("ipa", "snorm"), edge("snorm", "spatial"), edge("spatial", "sadd"),
            edge("ipa", "sadd", "skip", "residual"), edge("sadd", "summary"), edge("summary", "tnorm"),
            edge("tnorm", "temporal"), edge("temporal", "tadd"), edge("summary", "tadd", "skip", "residual"),
            edge("tadd", "fnorm"), edge("fnorm", "fc1"), edge("fc1", "fc2"), edge("fc2", "fadd"),
            edge("tadd", "fadd", "skip", "residual"), edge("time", "snorm", "L 參數", "condition"),
            edge("time", "tnorm", "T 參數", "condition"), edge("time", "fnorm", "MLP 參數", "condition")], [
                "三個 gated 子層：x ← x + gate · F((1 + scale) · LN(x) + shift)。gate 與 shift/scale 都來自 t。",
                "每層順序：可選 IPA → 空間注意力 → 可選摘要交流 → 時間混合 → MLP。",
                "虛線框節點關閉時直接略過；Hyena 與時間 Attention 二選一。prepend_ipa 與 interleave_ipa 可同時啟用。",
            ]),
        chart("03-prepend-ipa", "03 · 前置 IPALayer：只編碼構形",
              "這個 encoder 先產生 [B,L,C] 幾何條件，再廣播到所有影格。", [
            node("x", "Condition features", "胺基酸或端點相對變換", "[B,L,C]", "由 run_ipa 初始化。sim 使用胺基酸 embedding；TPS 等情境另外編碼起終點之間的相對框架。", LM, "def run_ipa(", 0, 0, "input"),
            node("norm", "ipa_norm", "正規化構形特徵", "[B,L,C]", "帶可學習 affine 的 LayerNorm，提供 IPA 輸入。", LM, "self.ipa_norm = nn.LayerNorm", 1, 0, "norm"),
            node("ipa", "Invariant Point Attention", "依三維位置彙整資訊", "[B,L,C]", "以 frames 轉換 query/key/value 空間點；local_spatial 時改為 run_local_ipa。詳見視圖 04。", LM, "x = x + self.ipa(self.ipa_norm(x), frames,", 2, 0, "geometry"),
            node("add1", "Residual addition", "加入幾何更新", "x + IPA(LN(x))", "IPA 的殘差沒有 AdaLN gate。", LM, "x = x + self.ipa(self.ipa_norm(x), frames,", 3, 0, "add"),
            node("frames", "Rigid frames", "提供旋轉與平移座標", "[B,L]", "這些是條件構形的框架，IPA 在這裡更新特徵，沒有直接修改 frame 座標。", LM, "frames=start_frames, partitions=partitions", 4, 0, "geometry"),
            node("snorm", "LayerNorm + AdaLN", "調變空間注意力輸入", "[B,L,C]", "以 t 產生 shift_msa_l、scale_msa_l；主幹維度沒有 T。", LM, "x = modulate(self.mha_layer_norm(x), shift_msa_l", 3, 1, "norm"),
            node("attn", "Spatial MHA / local MHA", "交流殘基之間的訊息", "[B,L,C]", "全域為 AttentionWithRoPE；局部為 LocalSelfAttention。", LM, "x = self.mha_l(x, mask=mask)", 2, 1, "spatial"),
            node("add2", "Residual + gate-L", "加入空間注意力更新", "[B,L,C]", "residual + gate_msa_l × attention_output。", LM, "x = residual + gate_msa_l.unsqueeze(1) * x", 1, 1, "add"),
            node("summary", "Cell summaries", "可選的全域摘要交流", "[B,L,C]", "只有 partitions 不為 None 才執行。", LM, "self.cell_summaries(x[batch_index:batch_index + 1], partition)", 0, 1, "spatial", True),
            node("fnorm", "LayerNorm + AdaLN", "調變 MLP 的輸入", "[B,L,C]", "使用 MLP 專用 shift 和 scale。", LM, "x = modulate(self.final_layer_norm(x), shift_mlp", 0, 2, "norm"),
            node("mlp", "fc1 → GELU → fc2", "逐殘基轉換通道", "C → 4C → C", "兩個 Linear 中間接 GELU，不在這裡混合時間。", LM, "x = self.fc2(gelu(self.fc1(x)))", 1, 2),
            node("add3", "Residual + gate-MLP", "輸出幾何條件特徵", "[B,L,C]", "此 IPALayer 重複 N 次之後，run_ipa 的輸出才以 [:,None] 廣播到 [B,T,L,C]。", LM, "x = residual + gate_mlp.unsqueeze(1) * x", 2, 2, "add"),
            node("time", "SiLU → Linear(C, 6C)", "產生兩組 AdaLN 參數", "2 × (shift, scale, gate)", "只有空間與 MLP 兩組；此層沒有時間注意力子層。", LM, "nn.Linear(self.embed_dim, 6 * self.embed_dim", 4, 2, "condition"),
        ], [edge("x", "norm"), edge("norm", "ipa"), edge("frames", "ipa", "座標", "condition"), edge("ipa", "add1"),
            edge("x", "add1", "skip", "residual"), edge("add1", "snorm"), edge("snorm", "attn"), edge("attn", "add2"),
            edge("add1", "add2", "skip", "residual"), edge("add2", "summary"), edge("summary", "fnorm"), edge("fnorm", "mlp"),
            edge("mlp", "add3"), edge("summary", "add3", "skip", "residual"),
            edge("time", "snorm", "空間參數", "condition"), edge("time", "fnorm", "MLP 參數", "condition")], [
                "前置 IPALayer：IPA → 空間 MHA → MLP；沒有 [B·L,T,C] 的時間混合。",
                "sim_condition 處理起始構形；TPS／inpainting 等以同一組 IPA 層分別處理兩個端點，再相加。",
                "這是 prepend_ipa 分支；主幹內的 interleave_ipa 是另一個獨立選項。",
            ]),
        chart("04-ipa", "04 · IPA 內部：把三維幾何變成注意力",
              "標量相似度與空間距離共同決定權重，再把結果轉回每個殘基的局部框架。", [
            node("s", "Single features s", "每個殘基的輸入特徵", "[...,L,C]", "省略號可代表 B 或 B,T。圖示為 MDGen 實際設定 c_z=0，因此沒有 pair-z 分支。", IPA, "q = self.linear_q(s)", 0, 1, "input"),
            node("qkv", "linear_q / linear_kv", "投影標量 Q、K、V", "[...,L,H,d_h]", "q 和 kv 分開線性投影，再按 head 整理；kv 分成 k 與 v。", IPA, "kv = self.linear_kv(s)", 1, 1),
            node("dot", "Scaled Q · K^T", "計算殘基特徵相似度", "[...,H,L,L]", "標量點積乘 sqrt(1/(3*d_h))；K^T 表示 K 的轉置。", IPA, "a *= math.sqrt", 2, 1, "spatial"),
            node("point", "Point projections", "產生局部 Q、K、V 空間點", "[...,L,H,P,3]", "linear_q_points 和 linear_kv_points 將特徵映射到多個三維點。Pq 與 Pv 可不同。", IPA, "q_pts = self.linear_q_points", 0, 0, "geometry"),
            node("frames", "Rigid.apply", "把局部點移到共同座標系", "R · point + translation", "對 Q/K/V 空間點使用各自殘基的剛體座標框架。", IPA, "q_pts = r[..., None].apply", 1, 0, "geometry"),
            node("distance", "Point distance score", "距離越遠，注意力分數越低", "[...,H,L,L]", "逐 Q/K 點計算平方距離，乘正的 head_weights 後加總，再乘 -0.5。", IPA, "pt_att = torch.sum(pt_att, dim=-1) * (-0.5)", 2, 0, "geometry"),
            node("sum", "Scalar + geometry + mask", "合併特徵相似度與距離", "[...,H,L,L]", "可加入 frame_mask 與 attn_mask 排除無效位置；MDGen c_z=0 時不使用 pair bias。", IPA, "a = a + pt_att", 3, 1, "add"),
            node("softmax", "Softmax → Dropout", "把分數轉成注意力權重", "[...,H,L,L]", "沿 key 殘基方向 softmax；訓練時可 dropout。", IPA, "a = self.softmax(a)", 4, 1, "spatial"),
            node("scalarout", "Weighted scalar V", "加權彙整標量內容", "[...,L,H·d_h]", "使用相同注意力矩陣對標量 v 加權。", IPA, "o = torch.matmul(a, v.transpose", 4, 2, "spatial"),
            node("pointout", "Weighted point V", "加權彙整全域空間點", "[...,L,H,Pv,3]", "v_pts 已透過各自框架轉入全域座標，先用注意力權重彙整。", IPA, "o_pt = torch.sum(", 4, 0, "geometry"),
            node("inverse", "Rigid.invert_apply", "轉回 query 殘基的座標系", "xyz + point norm", "結果轉回局部框架，再計算每個點的範數；這些是可學習的特徵，不是新原子座標。", IPA, "o_pt = r[..., None, None].invert_apply", 3, 0, "geometry"),
            node("concat", "Concat → linear_out", "融合標量與局部點特徵", "H·(d_h + 4Pv) → C", "串接標量、點的 x/y/z 分量與範數，再投影回 C。MDGen 的 c_z=0。", IPA, "concat_out_dim = self.no_heads", 3, 2),
            node("out", "IPA feature update", "回傳殘基特徵更新量", "[...,L,C]", "IPA 自己回傳更新量；外層 IPALayer 或 LatentMDGenLayer 才做 residual addition。", IPA, "return s", 2, 2, "input"),
        ], [edge("s", "qkv"), edge("qkv", "dot", "Q / K"), edge("s", "point"), edge("point", "frames"),
            edge("frames", "distance", "Q / K points"), edge("distance", "sum"), edge("dot", "sum"),
            edge("sum", "softmax"), edge("softmax", "scalarout"), edge("qkv", "scalarout", "V", "condition"),
            edge("softmax", "pointout"), edge("frames", "pointout", "V points", "condition"),
            edge("pointout", "inverse"), edge("inverse", "concat"), edge("scalarout", "concat"), edge("concat", "out")], [
                "MDGen 的 IPA 設定：c_z=0；一般 IPA 類別雖支援 pair feature，但此模型沒有使用該分支。",
                "H 注意力頭、d_h 每頭標量維度、Pq/Pv 每頭 QK／V 點數；預設 H=4、d_h=32、Pq=Pv=8。",
                "整體旋轉平移下，局部框架與距離讓特徵更新保持不變；IPA 並不直接移動蛋白質座標。",
            ]),
        chart("05-hyena", "05 · Hyena：沿時間軸的長卷積",
              "上排產生濾波器；中下排處理輸入。此處畫 MDGen 使用的 order=2。", [
            node("input", "Temporal sequence", "同一殘基的完整時間序列", "[B·L,T,C]", "外層先轉置 T/L；這裡把 B·L 視為一個批次維度。", HYENA, "def forward(self, u, *args", 0, 1, "input"),
            node("proj", "in_proj", "展開成三組通道", "C → 3C", "order=2，因此 inner_width = C*(order+1)=3C。", HYENA, "self.in_proj = nn.Linear", 1, 1),
            node("conv", "Depthwise Conv1d", "短範圍的時間濾波", "[B·L,3C,T]", "轉置成 channel-first，使用 kernel_size=3、padding=2、groups=3C 的深度卷積，裁切回 l_filter。", HYENA, "self.short_filter = nn.Conv1d", 2, 1, "temporal"),
            node("split", "Split: x0, x1, v", "分出兩個閘門與內容", "3 × [B·L,C,T]", "以每 C 通道切開。order=2 時是 x0、x1、v 三組。", HYENA, "*x, v = uc.split", 3, 1, "norm"),
            node("gate1", "Dropout(v × x1)", "第一個逐元素閘門", "[B·L,C,T]", "先將內容 v 乘 x1，再 dropout。", HYENA, "v = self.dropout(v * x_i)", 4, 1, "add"),
            node("fft", "fftconv(v, k, Ddiag)", "長卷積加逐通道捷徑", "FFT → multiply → IFFT", "以 2*T FFT 長度做卷積，裁切程式指定區段，另加 v*Ddiag。Ddiag 是學習到的每通道值，不是潛在維度 D。", HYENA, "out = y + u_fft * diagonal.unsqueeze(-1)", 4, 2, "temporal"),
            node("gate0", "v × x0", "第二個逐元素閘門", "[B·L,C,T]", "長卷積後乘 x0，再轉回 batch-time-channel。", HYENA, "y = rearrange(v * x[0]", 3, 2, "add"),
            node("outproj", "out_proj", "投影回時間特徵", "C → C", "Linear 在最後一維作用。", HYENA, "y = self.out_proj(y)", 2, 2),
            node("out", "Temporal update", "回到外層時間殘差", "[B·L,T,C]", "LatentMDGenLayer 外層會還原 [B,T,L,C]，乘 gate_T 並加上殘差。", HYENA, "y = self.out_proj(y)", 1, 2, "input"),
            node("pos", "PositionalEmbedding", "為濾波器編碼時間位置", "[1,T,3]", "filter_fn 的位置編碼不是直接加入輸入序列 u；它只用來產生長卷積核。", HYENA, "z, t = self.pos_emb(L)", 1, 0, "condition"),
            node("filter", "Implicit filter MLP", "學習每個時間距離的濾波器", "3 → 64 → 64 → 64 → C", "各 hidden Linear 後為 Sin，最後 Linear 無 bias。此處為 MDGen filter_order=64 與預設 num_inner_mlps=2。", HYENA, "self.implicit_filter = nn.Sequential", 2, 0),
            node("decay", "ExponentialModulation", "控制長距離的衰減", "k: [C,T]", "以 exp(-t*abs(delta)) 調變隱式濾波器，再重排成 order 與通道維度。", HYENA, "h = self.modulation(t, h)", 3, 0, "temporal"),
            node("diag", "filter_fn.bias", "可學習的通道捷徑係數", "Ddiag: [C]", "這個名為 bias 的參數在 fftconv 中用於乘上輸入的對角項。", HYENA, "self.bias = nn.Parameter", 4, 0, "condition"),
        ], [edge("input", "proj"), edge("proj", "conv"), edge("conv", "split"), edge("split", "gate1", "v, x1"),
            edge("gate1", "fft"), edge("fft", "gate0"), edge("split", "gate0", "x0", "condition"), edge("gate0", "outproj"),
            edge("outproj", "out"), edge("pos", "filter"), edge("filter", "decay"), edge("decay", "fft", "k", "condition"),
            edge("diag", "fft", "Ddiag", "condition")], [
                "--hyena 只替換主幹的時間注意力；空間注意力與 IPA 的選項仍各自獨立。",
                "圖示 order=2、filter_order=64；軌跡 T 必須符合設定的 num_frames，主幹 Hyena 分支要求 mask 全為有效。",
                "小立方體僅表達模組與資料流，體積不代表參數量、通道數比例或運算耗時。",
            ]),
        chart("06-local-spatial", "06 · 局部分區與全域摘要",
              "局部 attention 使用 core + ghost；只有 core 寫回。摘要在所有區域之間交換。", [
            node("coords", "start_frames / chain IDs", "條件座標與蛋白鏈資訊", "[B,L,3] + masks", "LatentMDGenModel 以起始條件座標建立分區；不是在每個 layer 自動重新分區。", LM, "build_spatial_partition(", 0, 0, "geometry"),
            node("partition", "SpatialPartition", "定義唯一核心歸屬與鄰域", "G groups; Q = core + ghost", "每個有效殘基只有一個 core owner；ghost 提供唯讀的局部邊界脈絡。", "mdgen/full_length/partition.py", "class SpatialPartition:", 1, 0, "geometry"),
            node("x", "Modulated spatial features", "單一蛋白的所有影格特徵", "[T,L,C]", "圖從空間 AdaLN 之後開始。B 維由外層逐個 batch 處理。", LOCAL, "def forward(", 0, 1, "input"),
            node("gather", "Gather core + ghost", "收集每個局部鄰域", "[T,G,Q,C]", "gather 先得到 [G,Q,T,C]，再 permute 到 [T,G,Q,C]。", LOCAL, "grouped = gathered.values.permute", 1, 1, "geometry"),
            node("local", "QKV → SDPA / Flex", "各群內的注意力計算", "[T·group_batch,Q,C]", "以 group_batch_size 分批；SDPA 與 FlexAttention 是同一局部 attention 的後端選項。", LOCAL, "outputs[:, start:stop] = self._attend", 2, 1, "spatial"),
            node("scatter", "scatter_core", "只寫回核心殘基", "[T,L,C]", "包含 ghost 的局部結果不會全部加回；只有 core owner 輸出寫回全域位置。", LOCAL, "assembled = partition.scatter_core", 3, 1, "geometry"),
            node("add", "Spatial residual + gate", "與子層前的特徵相加", "[T,L,C]", "使用空間子層輸入 residual（在 AdaLN 之前）加 gate_L × local_update。圖中省略該外側 skip 線。", LM, "x = residual + gate_msa_l.unsqueeze(1) * x", 4, 1, "add"),
            node("pool", "pool_core", "每區只平均核心特徵", "[T,G,C]", "重新依分區 gather，再用 core_mask 做平均；ghost 不進摘要平均。", SUMMARY, "summaries = self.pool_core", 4, 2, "norm"),
            node("global", "Summary MHA", "所有區域摘要互相注意", "G × G per frame", "nn.MultiheadAttention 在所有 G 個摘要間計算；不是只傳送給相鄰區域。", SUMMARY, "exchanged, _ = self.summary_attention", 3, 2, "spatial"),
            node("norm", "Residual → LayerNorm", "整合原摘要與交換結果", "[T,G,C]", "summary_norm(summaries + exchanged)。", SUMMARY, "exchanged = self.summary_norm", 2, 2, "norm"),
            node("inject", "Owner lookup → Linear", "把區域脈絡給所屬殘基", "[T,G,C] → [T,L,C]", "依 owner_group 選出每個殘基的摘要，再 Linear inject，最後加回 states。", SUMMARY, "context[:, valid] = exchanged[:, owner[valid]]", 1, 2),
            node("out", "states + injected", "接回時間混合子層", "[T,L,C]", "回到 LatentMDGenLayer 後堆回 B 維，接續時間 AdaLN 和時間 Attention/Hyena。", SUMMARY, "return states + injected", 0, 2, "input"),
        ], [edge("coords", "partition"), edge("partition", "gather", "core / ghost", "condition"), edge("x", "gather"),
            edge("gather", "local"), edge("local", "scatter"), edge("scatter", "add"), edge("add", "pool"), edge("pool", "global"),
            edge("global", "norm"), edge("norm", "inject"), edge("inject", "out"), edge("add", "out", "skip", "residual")], [
                "G 是區域數，Q 是各局部群的 core + ghost 容量（含 padding）；圖中沿空間分組，並非裁掉整段序列。",
                "Local IPA 採同樣的 gather → 群內 IPA → scatter_core 規則；本頁展開的是空間 self-attention + 摘要。",
                "主模型 local 分支沒有直接呼叫 BoundedTrajectoryBlock；下一頁是獨立的規模驗證模型。",
            ]),
        chart("07-scale-block", "07 · 規模驗證用 BoundedTrajectoryBlock",
              "獨立的工程量測模型：空間 attention → 摘要 → local IPA → Hyena → 位移頭。", [
            node("input", "states + frames + partition", "一個蛋白的測試輸入", "states: [1,T,L,C]", "目前 forward 限制 B=1。frames 是每個影格的 Rigid，partition 提供 core/ghost 分區。", PIPE, "if states.shape[0] != 1:", 0, 0, "input"),
            node("snorm", "spatial_norm", "正規化空間輸入", "[T,L,C]", "先 states[0] 取出單一分子的軌跡；時間分片由 _map_frame_chunks 管理。", PIPE, "self.spatial_norm(value)", 1, 0, "norm"),
            node("spatial", "LocalSelfAttention", "分組計算空間注意力", "[T,L,C]", "每批只處理部分群組，只有 core 更新回填。", PIPE, "lambda value, _start, _stop: self.spatial(", 2, 0, "spatial"),
            node("add1", "Spatial residual", "加入空間更新", "trajectory + spatial_update", "直接加回殘差；本驗證 block 沒有生成時間 t 的 AdaLN gate。", PIPE, "trajectory = trajectory + self._map_frame_chunks(", 3, 0, "add"),
            node("summary", "CoreCellSummaryExchange", "交換全域區域摘要", "[T,L,C]", "此模組內部已做 states + injected；外層沒有再多加一次相同殘差。", PIPE, "lambda value, _start, _stop: self.summaries", 4, 0, "spatial"),
            node("ipanorm", "ipa_norm", "正規化幾何輸入", "[T,L,C]", "幾何子層跟在摘要交換後。", PIPE, "self.ipa_norm(value)", 4, 1, "norm"),
            node("ipa", "Local IPA", "各影格的局部幾何更新", "[T,L,C]", "使用對應 frame chunk 的 frames[start:stop]。", PIPE, "lambda value, start, stop: self.local_ipa(", 3, 1, "geometry"),
            node("add2", "Geometry residual", "加入幾何更新", "trajectory + IPA_update", "完成後 unsqueeze 恢復 [1,T,L,C]。", PIPE, "states = trajectory.unsqueeze(0)", 2, 1, "add"),
            node("tnorm", "temporal_norm", "正規化時間輸入", "[1,T,L,C]", "LayerNorm 在時間 mixer 之前。", PIPE, "self.temporal_norm(states)", 1, 1, "norm"),
            node("temporal", "TemporalMixer: Hyena", "分批處理殘基時間序列", "[L,T,C] → [1,T,L,C]", "此 block 固定 backend=hyena。沿 B·L 切批，每個殘基仍保留完整 T；可做 activation checkpoint。", PIPE, 'backend="hyena"', 0, 1, "temporal"),
            node("add3", "Temporal residual", "輸出更新後特徵", "[1,T,L,C]", "states + temporal(norm(states))。這些 states 與位移會一併回傳。", PIPE, "states = states + self.temporal(", 0, 2, "add"),
            node("head", "coordinate_head", "測試用的三維位移投影", "Linear: C → 3", "這是獨立驗證 block 的小型頭，不是 LatentMDGenModel 的 FinalLayer。", PIPE, "self.coordinate_head = nn.Linear", 1, 2),
            node("output", "0.01 × displacement", "輸出測試用位移", "[1,T,L,1,3]", "coordinate_head 輸出乘 0.01 並 unsqueeze(-2)。用途是規模／幾何驗證，未宣稱此頭是訓練好的分子生成模型。", PIPE, "displacement = 0.01", 2, 2, "input"),
        ], [edge("input", "snorm"), edge("snorm", "spatial"), edge("spatial", "add1"), edge("input", "add1", "skip", "residual"),
            edge("add1", "summary"), edge("summary", "ipanorm"), edge("ipanorm", "ipa"), edge("ipa", "add2"),
            edge("summary", "add2", "skip", "residual"), edge("add2", "tnorm"), edge("tnorm", "temporal"),
            edge("temporal", "add3"), edge("add2", "add3", "skip", "residual"), edge("add3", "head"), edge("head", "output")], [
                "這是 scale_validation 使用的工程 block，沒有串在 LatentMDGenModel.forward 的正式生成路徑中。",
                "空間階段按影格切片，時間階段按殘基序列切批；checkpoint 以重算節省反向傳播的記憶體。",
                "圖示忠實區分兩套程式的層順序；規模驗證通過本身不等於長蛋白動力學已獲科學驗證。",
            ]),
    ]


def route_edges(graph):
    """Orthogonal routes on a small grid, avoiding all layer rectangles."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    blocked = set()
    for n in nodes.values():
        for x in range(n["x"] - 10, n["x"] + NW + 21, 10):
            for y in range(n["y"] - 30, n["y"] + NH + 11, 10):
                blocked.add((x, y))
    used = set()

    def port(n, side):
        x, y = n["x"], n["y"]
        return {"r": ((x + NW, y + 50), (x + NW + 30, y + 50)),
                "l": ((x, y + 50), (x - 20, y + 50)),
                "t": ((x + 120, y - 16), (x + 120, y - 40)),
                "b": ((x + 120, y + NH), (x + 120, y + NH + 20))}[side]

    for e in graph["edges"]:
        a, b = nodes[e["source"]], nodes[e["target"]]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        if e["kind"] == "residual":
            sa, sb = ("t", "t") if dy == 0 else ("b", "b")
        elif abs(dx) >= abs(dy):
            sa, sb = ("r", "l") if dx > 0 else ("l", "r")
        else:
            sa, sb = ("b", "t") if dy > 0 else ("t", "b")
        begin, start = port(a, sa)
        end, goal = port(b, sb)
        initial = (*start, -1)
        queue = [(0, 0, initial)]
        best, previous = {initial: 0}, {}
        found = None
        while queue:
            _, cost, state = heapq.heappop(queue)
            x, y, direction = state
            if cost != best[state]:
                continue
            if (x, y) == goal:
                found = state
                break
            for d, (vx, vy) in enumerate(((10, 0), (-10, 0), (0, 10), (0, -10))):
                nx, ny = x + vx, y + vy
                if not (20 <= nx <= 1610 and 125 <= ny <= 870):
                    continue
                if (nx, ny) in blocked and (nx, ny) != goal:
                    continue
                nxt = (nx, ny, d)
                nc = cost + 10 + (12 if d != direction and direction != -1 else 0) + (18 if (nx, ny) in used else 0)
                if nc < best.get(nxt, float("inf")):
                    best[nxt], previous[nxt] = nc, state
                    h = abs(nx - goal[0]) + abs(ny - goal[1])
                    heapq.heappush(queue, (nc + h, nc, nxt))
        if found is None:
            raise RuntimeError(f"Cannot route {graph['id']}: {e}")
        path = []
        while found != initial:
            path.append(found[:2])
            found = previous[found]
        path.append(start)
        path.reverse()
        used.update(path)
        points = [begin, *path, end]
        reduced = [points[0]]
        for i in range(1, len(points) - 1):
            p, q, r = reduced[-1], points[i], points[i + 1]
            if (q[0] - p[0]) * (r[1] - q[1]) != (q[1] - p[1]) * (r[0] - q[0]):
                reduced.append(q)
        reduced.append(points[-1])
        e["points"] = reduced
        e["ports"] = [sa, sb]


def render_static(graph, pdf):
    fig, ax = plt.subplots(figsize=(16.4, 10), dpi=140)
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.text(50, 43, graph["title"], fontsize=24, weight="bold", color="#172b42", va="center")
    ax.text(50, 82, graph["subtitle"], fontsize=13, color="#536379", va="center")
    ax.text(50, 111, "實線：資料流　 ··· 條件控制　 --- 殘差跳接　 虛線框：可選模組", fontsize=10, color="#697990")
    for e in graph["edges"]:
        color = {"residual": "#a85553", "condition": "#387e76", "optional": "#79588b"}.get(e["kind"], "#5b6c80")
        ls = {"residual": "--", "condition": ":", "optional": "--"}.get(e["kind"], "-")
        pts = e["points"]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=1.6, linestyle=ls, zorder=1)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.6, zorder=2))
        if e["label"]:
            segments = [(abs(b[0]-a[0])+abs(b[1]-a[1]), a, b) for a, b in zip(pts, pts[1:])]
            _, a, b = max(segments)
            ax.text((a[0]+b[0])/2, (a[1]+b[1])/2 - 4, e["label"], fontsize=8, color=color,
                    ha="center", va="bottom", bbox=dict(facecolor="#f7f9fc", edgecolor="none", pad=1), zorder=3)
    for i, n in enumerate(graph["nodes"]):
        x, y, color = n["x"], n["y"], LEGEND[n["kind"]]
        ls = "--" if n["optional"] else "-"
        faces = [Polygon([(x,y),(x+18,y-16),(x+NW+18,y-16),(x+NW,y)], facecolor="white"),
                 Polygon([(x+NW,y),(x+NW+18,y-16),(x+NW+18,y+NH-16),(x+NW,y+NH)], facecolor=color["stroke"], alpha=.45),
                 Rectangle((x,y), NW, NH, facecolor=color["fill"])]
        for j, face in enumerate(faces):
            face.set_edgecolor(color["stroke"])
            face.set_linewidth(1.15)
            face.set_linestyle(ls)
            face.set_gid(f"node-{n['id']}-face{j}")
            face.set_zorder(4)
            ax.add_patch(face)
        label = "\n".join(textwrap.wrap(n["label"], width=31, break_long_words=False))
        a = ax.text(x+12, y+15, label, fontsize=10.5, weight="bold", color="#233449", va="top", zorder=5)
        a.set_gid(f"node-{n['id']}-label")
        a = ax.text(x+12, y+58, n["zh"], fontsize=11, color="#263c51", va="top", zorder=5)
        a.set_gid(f"node-{n['id']}-zh")
        shape = n["shape"]
        a = ax.text(x+12, y+89, shape, fontsize=8.4, color="#4e5c6b", va="top", zorder=5)
        a.set_gid(f"node-{n['id']}-shape")
    for i, note in enumerate(graph["notes"]):
        ax.text(50, 906+i*25, note, fontsize=10.1, color="#536379", va="center")
    ax.text(1580, 982, "MDGen · source-grounded architecture", fontsize=8.3, ha="right", color="#8593a4")
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", facecolor=fig.get_facecolor())
    svg_text = buffer.getvalue()
    svg_text = svg_text[svg_text.index("<svg"):]
    # Group all artists of a layer for click and focus interactions in the viewer.
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.fromstring(svg_text)
    for n in graph["nodes"]:
        group = ET.Element("{http://www.w3.org/2000/svg}g", {"id": f"node-{n['id']}", "role": "button", "tabindex": "0", "aria-label": n["zh"]})
        for parent in list(root.iter()):
            for child in list(parent):
                if child.get("id", "").startswith(f"node-{n['id']}-"):
                    parent.remove(child)
                    group.append(child)
        root.append(group)
    svg_text = ET.tostring(root, encoding="unicode")
    (OUT / f"{graph['id']}.svg").write_text(svg_text)
    graph["svg"] = svg_text
    fig.savefig(OUT / f"{graph['id']}.png", dpi=140, facecolor=fig.get_facecolor())
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_drawio(graphs):
    root = ET.Element("mxfile", host="MDGen", version="26.0.0")
    ports = {"r": (1, .5), "l": (0, .5), "t": (.5, 0), "b": (.5, 1)}
    for graph in graphs:
        diagram = ET.SubElement(root, "diagram", name=graph["title"], id=graph["id"])
        model = ET.SubElement(diagram, "mxGraphModel", page="1", pageWidth=str(WIDTH), pageHeight=str(HEIGHT))
        cells = ET.SubElement(model, "root")
        ET.SubElement(cells, "mxCell", id="0")
        ET.SubElement(cells, "mxCell", id="1", parent="0")
        def vertex(id, value, style, x, y, w, h):
            cell = ET.SubElement(cells, "mxCell", id=id, value=value, style=style, vertex="1", parent="1")
            ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
        vertex("title", graph["title"], "text;html=1;fontSize=26;fontStyle=1;align=left;", 50, 20, 1500, 50)
        vertex("subtitle", graph["subtitle"], "text;html=1;fontSize=14;align=left;", 50, 80, 1500, 40)
        for n in graph["nodes"]:
            c = LEGEND[n["kind"]]
            value = f"<b>{html.escape(n['label'])}</b><br>{html.escape(n['zh'])}<br>{html.escape(n['shape'])}"
            style = f"shape=cube;whiteSpace=wrap;html=1;fontSize=13;fillColor={c['fill']};strokeColor={c['stroke']};"
            if n["optional"]:
                style += "dashed=1;"
            vertex(n["id"], value, style, n["x"], n["y"], NW, NH)
        for i, e in enumerate(graph["edges"]):
            a, b = (ports[p] for p in e["ports"])
            color = {"condition": "#387e76", "residual": "#a85553", "optional": "#79588b"}.get(e["kind"], "#5b6c80")
            style = f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;strokeColor={color};"
            style += f"exitX={a[0]};exitY={a[1]};entryX={b[0]};entryY={b[1]};"
            if e["kind"] != "data":
                style += "dashed=1;"
            cell = ET.SubElement(cells, "mxCell", id=f"edge-{i}", parent="1", edge="1", source=e["source"], target=e["target"], value=e["label"], style=style)
            geom = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
            points = ET.SubElement(geom, "Array", **{"as": "points"})
            for x, y in e["points"][1:-1]:
                ET.SubElement(points, "mxPoint", x=str(x), y=str(y))
        for i, note in enumerate(graph["notes"]):
            vertex(f"note-{i}", note, "text;html=1;fontSize=13;align=left;", 50, 900+i*30, 1500, 30)
    ET.ElementTree(root).write(OUT / "mdgen-architecture.drawio", encoding="utf-8", xml_declaration=True)


def validate_bundle(graphs):
    sources = {}
    for graph in graphs:
        ids = [n["id"] for n in graph["nodes"]]
        assert len(ids) == len(set(ids)), graph["id"]
        for n in graph["nodes"]:
            assert re.search(r"[\u3400-\u9fff]", n["zh"]), n["id"]
            assert n["shape"] and n["description"]
            file = n["source"]["file"]
            source = (ROOT / file).read_text()
            assert n["source"]["anchor"] in source.splitlines()[n["source"]["line"] - 1]
            sources[file] = hashlib.sha256(source.encode()).hexdigest()
        for e in graph["edges"]:
            assert e["source"] in ids and e["target"] in ids
            assert len(e["points"]) >= 2
        root = ET.fromstring(graph["svg"])
        svg_ids = {el.get("id") for el in root.iter()}
        assert all(f"node-{id}" in svg_ids for id in ids)
    report = dict(charts=len(graphs), nodes=sum(len(g["nodes"]) for g in graphs),
                  edges=sum(len(g["edges"]) for g in graphs),
                  source_anchors="all matched", svg_node_groups="all matched",
                  source_sha256=sources, runtime_weights_loaded=False,
                  tool_docs=["https://plotly.com/python/3d-mesh/", "https://plotly.com/javascript/plotlyjs-events/"])
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    font = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["svg.fonttype"] = "none"
    graphs = charts()
    with PdfPages(OUT / "mdgen-architecture.pdf") as pdf:
        for graph in graphs:
            route_edges(graph)
            render_static(graph, pdf)
            print(f"Rendered {graph['id']}: {len(graph['nodes'])} layers", flush=True)
    bundle = dict(charts=graphs, legend=LEGEND, built_at=datetime.now(timezone.utc).isoformat(),
                  scope="目前工作樹的程式架構；非權重分析或實際 activation 測量。")
    (OUT / "architecture-data.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2))
    write_drawio(graphs)
    print(json.dumps(validate_bundle(graphs), ensure_ascii=False), flush=True)
    template = OUT / "viewer.html"
    if template.exists():
        content = template.read_text().replace("__PLOTLY_JS__", get_plotlyjs())
        content = content.replace("__ARCHITECTURE_DATA__", json.dumps(bundle, ensure_ascii=False).replace("</", "<\\/"))
        (OUT / "index.html").write_text(content)
        print("Built offline index.html", flush=True)


if __name__ == "__main__":
    main()
