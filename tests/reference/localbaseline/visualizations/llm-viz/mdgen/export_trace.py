"""Export an actual tiny MDGen layer and verify its operator-level replay.

No checkpoints or training code are modified. Random weights are explicitly
labelled; no physical interpretation should be attached to their values.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent))
import torch
from mdgen.model.latent_model import LatentMDGenLayer
from mdgen.model.layers import gelu, modulate

DEST = Path(__file__).resolve().parent


def matrix(name, tensor, row="位置", col="特徵", kind="i"):
    tensor = tensor.detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor[None]
    assert tensor.ndim == 2 and torch.isfinite(tensor).all()
    return dict(name=name, values=tensor.tolist(), row=row, col=col, kind=kind)


def attention(module, x):
    """Replay the unmasked, non-RoPE branch, including the learned bias KV row."""
    a = module.attn
    batch, length, channels = x.shape
    heads, d = a.num_heads, a.head_dim
    q, k, v = a.q_proj(x), a.k_proj(x), a.v_proj(x)
    if a.bias_k is not None:
        k = torch.cat([k, a.bias_k.expand(batch, 1, channels)], 1)
        v = torch.cat([v, a.bias_v.expand(batch, 1, channels)], 1)
    qh = q.reshape(batch, length, heads, d).transpose(1, 2)
    kh = k.reshape(batch, -1, heads, d).transpose(1, 2)
    vh = v.reshape(batch, -1, heads, d).transpose(1, 2)
    score = (qh * a.scaling) @ kh.transpose(-1, -2)
    probs = score.softmax(-1)
    mixed = probs @ vh
    merged = mixed.transpose(1, 2).reshape(batch, length, channels)
    out = a.out_proj(merged)
    actual = module(x, torch.ones(batch, length))
    torch.testing.assert_close(out, actual, atol=2e-6, rtol=2e-5)
    return dict(q=qh, k=kh, v=vh, score=score, probs=probs,
                mixed=mixed, merged=merged, out=out)


@torch.no_grad()
def export():
    torch.set_num_threads(1)
    torch.manual_seed(20260906)
    B, T, L, C, H = 1, 3, 4, 8, 2
    layer = LatentMDGenLayer(C, 4*C, H, num_frames=T).eval()
    x = torch.randn(B, T, L, C)
    t = torch.randn(B, 1, C)  # Already-embedded flow time, not a trajectory index.
    actual = layer(x, t, mask=torch.ones(B, T, L))
    ada = layer.adaLN_modulation(t)
    sl, scl, gl, st, sct, gt, sm, scm, gm = ada.chunk(9, -1)
    norm_l = layer.mha_layer_norm(x)
    mod_l = modulate(norm_l, sl, scl)
    spatial = attention(layer.mha_l, mod_l.reshape(B*T, L, C))
    update_l = spatial['out'].reshape(B, T, L, C)
    after_l = x + gl.unsqueeze(1) * update_l
    norm_t = layer.mha_layer_norm(after_l)
    mod_t = modulate(norm_t, st, sct)
    reordered = mod_t.transpose(1, 2).reshape(B*L, T, C)
    temporal = attention(layer.mha_t, reordered)
    update_t = temporal['out'].reshape(B, L, T, C).transpose(1, 2)
    after_t = after_l + gt.unsqueeze(1) * update_t
    norm_m = layer.final_layer_norm(after_t)
    mod_m = modulate(norm_m, sm, scm)
    hidden = layer.fc1(mod_m)
    activated = gelu(hidden)
    update_m = layer.fc2(activated)
    replay = after_t + gm.unsqueeze(1) * update_m
    torch.testing.assert_close(replay, actual, atol=3e-6, rtol=2e-5)
    cases = []
    for time in range(T):
      for residue in range(L):
       for head in range(H):
        stages = []
        def add(title, note, mats, op="inspect", line=568, **extra):
            stages.append(dict(title=title, note=note, matrices=mats, op=op,
                source=f"mdgen/model/latent_model.py:{line}", **extra))
        def m(name, a, row="殘基", col="特徵", kind="i"):
            return matrix(name, a, row, col, kind)
        def norms(prefix, before, norm, mod, shift, scale, line, idx):
            add(prefix+"：LayerNorm", "每個位置獨立沿 C 軸正規化，不混合時間或殘基。點右側輸出格看該列的平均與變異數。",
                [m("輸入", before[idx]), m("正規化", norm[idx])], "norm", line, eps=1e-6)
            add(prefix+"：AdaLN", "流時間條件產生 scale 與 shift；它們沿時間與殘基廣播。t 不等於軌跡的 T。",
                [m("正規化", norm[idx]), m("scale", scale[0, 0],kind="w"), m("shift",shift[0,0],kind="w"),m("調變後",mod[idx])], "modulate",line)
        add("輸入：時間 × 殘基 × 特徵", "縮小版真實 forward：B=1、T=3、L=4、C=8、2 個 head。此處展示所選時間的 L×C 切片；可切換時間、殘基與 head。",[m("x：所選時間", x[0,time])])
        add("條件分流：9 組 AdaLN 參數", "SiLU(t) 經 Linear(C,9C)，切成空間、時間、MLP 各自的 shift／scale／gate；gate 不是注意力機率，可以為負值。",[m("時間條件 t", t[0]),m("九組參數",ada.reshape(9,C),"參數組")],line=571)
        norms("空間",x,norm_l,mod_l,sl,scl,597,(0,time))
        def attn_stages(label, a, inp, mod, group, axis, line):
            d=C//H; span=slice(head*d,(head+1)*d)
            for key in ['q','k','v']:
                proj=getattr(mod.attn,key+'_proj')
                raw=proj(inp)[group,:,span]
                add(label+f"：{key.upper()} 投影", "每列特徵乘上權重，再加 bias；顯示所選 head 的 4 個通道。點輸出格可展開逐項乘加。",
                    [m("輸入",inp[group],axis),m("Wᵀ",proj.weight[span].T,"輸入特徵","head 特徵","w"),m("bias",proj.bias[span],kind="w"),m(key.upper(),raw,axis,"head 特徵")],"linear",line)
            add(label+"：額外的 bias K／V", "原始 layer 啟用 add_bias_kv：K、V 最後多一列可學習記憶，不是額外殘基或時間點。Q 不增加列。",[m("K（含 bias 列）",a['k'][group,head],axis+"＋bias","head 特徵"),m("V（含 bias 列）",a['v'][group,head],axis+"＋bias","head 特徵")],line=line)
            add(label+"：QKᵀ / √d", "選右側分數格，會高亮對應的 Q 列與 Kᵀ 欄；所有乘積加總後除以 √4。這是相似度，還不是機率。",[m("Q",a['q'][group,head],axis,"head 特徵"),m("Kᵀ",a['k'][group,head].T,"head 特徵",axis+"＋bias"),m("分數",a['score'][group,head],axis,axis+"＋bias","a")],"matmul",line,factor=mod.attn.scaling)
            add(label+"：Softmax", "每個 query 的整列分數轉成總和為 1 的權重。此示範全部位置有效，沒有 causal mask；不要套用 GPT 的下三角遮罩。",[m("分數",a['score'][group,head],axis,axis+"＋bias"),m("注意力權重",a['probs'][group,head],axis,axis+"＋bias","a")],"softmax",line)
            add(label+"：加權混合 V", "一個輸出格來自多個位置的 V 加權和。點輸出格，逐項看每個來源貢獻多少；不是把同一個數字送到下一層。",[m("注意力權重",a['probs'][group,head],axis,axis+"＋bias","a"),m("V",a['v'][group,head],axis+"＋bias","head 特徵"),m("head 輸出",a['mixed'][group,head],axis,"head 特徵")],"matmul",line,factor=1)
            add(label+"：合併 heads", "兩個 head 各有 4 維，依 head 順序串接為 8 維；不是把 heads 相加。",[m("head 1",a['mixed'][group,0],axis),m("head 2",a['mixed'][group,1],axis),m("串接",a['merged'][group],axis)],"concat",line)
            add(label+"：輸出投影", "合併後再經 out_proj 混合特徵；這個更新量還要經 gate 才加入殘差。",[m("串接",a['merged'][group],axis),m("Wᵀ",mod.attn.out_proj.weight.T,"輸入特徵","輸出特徵","w"),m("bias",mod.attn.out_proj.bias,kind="w"),m("更新量",a['out'][group],axis)],"linear",line)
        attn_stages("空間 Attention",spatial,mod_l.reshape(T,L,C),layer.mha_l,time,"殘基",599)
        def residual(label,before,update,gate,after,line):
            add(label,"原值保留，加上 gate × 更新量。請點最右側格子，比較原值、更新量和新值。",[m("原值",before[0,time]),m("更新量",update[0,time]),m("gate",gate[0,0],kind="w"),m("新值",after[0,time])],"residual",line)
        residual("空間殘差",x,update_l,gl,after_l,611)
        norms("時間",after_l,norm_t,mod_t,st,sct,619,(0,time))
        add("軸重排：現在沿時間讀取", "左邊每列是 (時間,殘基)，右邊每列是 (殘基,時間)。點右邊格子看它搬自哪裡：數值不變，分組方式改變。接下來固定所選殘基，對 3 個時間點做 Attention。",[m("T×L 排列",mod_t.reshape(T*L,C),"時間×殘基"),m("L×T 排列",reordered.reshape(L*T,C),"殘基×時間")],"reorder",626,T=T,L=L)
        attn_stages("時間 Attention",temporal,reordered,layer.mha_t,residue,"時間",626)
        residual("時間殘差（還原 T,L 軸）",after_l,update_t,gt,after_t,630)
        norms("MLP",after_t,norm_m,mod_m,sm,scm,633,(0,time))
        add("MLP：C → 4C", "每個位置獨立從 8 維展開為 32 維；不在此步混合殘基或時間。",[m("輸入",mod_m[0,time]),m("Wᵀ",layer.fc1.weight.T,"輸入特徵","隱藏特徵","w"),m("bias",layer.fc1.bias,kind="w"),m("展開",hidden[0,time])],"linear",634)
        add("MLP：GELU", "逐元素套用原始程式使用的 GELU（erf 版本），小的負值被抑制，但不是直接全部截成零。",[m("GELU 前",hidden[0,time]),m("GELU 後",activated[0,time])],"gelu",634)
        add("MLP：4C → C", "從 32 維投影回 8 維，才能與原本的殘差相加。",[m("隱藏特徵",activated[0,time]),m("Wᵀ",layer.fc2.weight.T,"隱藏特徵","輸出特徵","w"),m("bias",layer.fc2.bias,kind="w"),m("更新量",update_m[0,time])],"linear",634)
        residual("Layer 輸出",after_t,update_m,gm,replay,635)
        cases.append(dict(time=time,residue=residue,head=head,stages=stages))
    sources={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'model/latent_model.py',ROOT/'model/mha.py',ROOT/'model/layers.py']}
    result=dict(meta=dict(seed=20260906,shape=[B,T,L,C],heads=H,torch=torch.__version__,
        provenance="真實 MDGen LatentMDGenLayer；固定隨機初始化，未載入 checkpoint。",
        exclusions="本數值示範未啟用 IPA、Hyena、local_spatial、RoPE；不是完整模型推論。",
        max_error=(actual-replay).abs().max().item(),sources=sources),cases=cases)
    (DEST/'trace.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(cases=len(cases),stages=len(cases[0]['stages']),max_error=result['meta']['max_error']),indent=2))


if __name__ == '__main__':
    export()
