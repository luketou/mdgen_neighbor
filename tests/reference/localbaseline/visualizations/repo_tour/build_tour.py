#!/usr/bin/env python3
"""Source-only inventory and shared storyboards; never import the model."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXCLUDED = ('.omx/', '.scratch/', '.git/', 'mdgen/visualizations/', '__pycache__/')


def source(file, anchor, after=None):
    lines = (ROOT / file).read_text().splitlines()
    start = next((i for i, s in enumerate(lines) if after and after in s), 0)
    line = next(i + 1 for i in range(start, len(lines)) if anchor in lines[i])
    return {'file': file, 'line': line, 'anchor': anchor}


def node(id, zh, label, shape, description, file, anchor, kind='linear', optional=False):
    return dict(id=id, zh=zh, label=label, shape=shape, description=description,
                source=source(file, anchor), kind=kind, optional=optional)


def edge(a, b, label='', kind='data'):
    return dict(source=a, target=b, label=label, kind=kind)


def chapter(id, title, summary, nodes, edges, notes=()):
    return dict(id=id, title=title, summary=summary, kind='flow', nodes=nodes,
                edges=edges, notes=list(notes), steps=[dict(node=n['id'], explanation=n['description']) for n in nodes])


def neural_chapters():
    old = json.loads((HERE.parent / 'architecture-data.json').read_text())
    chapters = [chapter('00-repo-map','00 · 先看整個 repo 如何串起來',
        '先區分資料準備、訓練、生成與分析，再進入每一層；工程驗證是獨立支線。',[
        node('prep','把模擬軌跡轉成模型資料','scripts/prep_sims.py','trajectory → Atom14 .npy','下載或產生的軌跡經原子映射，寫出 Atom14 格式；split CSV 決定訓練與驗證樣本。','scripts/prep_sims.py','def traj_to_atom14','input'),
        node('dataset','載入與抽取軌跡片段','MDGenDataset','[T,L,14,3] → batch dict','dataset 負責時間取樣、幾何轉換與可選長度裁切；不負責訓練模型。','mdgen/dataset.py','class MDGenDataset','input'),
        node('train','啟動訓練迴圈','train.py / Lightning Trainer','DataLoader → training steps','train.py 組裝 loader、wrapper、logger 與 Trainer，並處理 checkpoint 恢復。','train.py','trainer = pl.Trainer(','condition'),
        node('wrapper','統一任務條件與目標','NewMDGenWrapper','batch → latent + model_kwargs','wrapper 連接資料與模型：建立 latent、遮罩、幾何條件，計算 loss，並處理推論重建。','mdgen/wrapper.py','class NewMDGenWrapper','linear'),
        node('model','學習空間與時間的更新','LatentMDGenModel','[B,T,L,D] → [B,T,L,D]','真正的神經主幹：條件編碼、IPA 選項、空間/時間混合、MLP 與輸出頭。','mdgen/model/latent_model.py','class LatentMDGenModel','spatial'),
        node('transport','定義流匹配與積分取樣','Transport / Sampler','t、x(t)、velocity / score / noise','訓練建立插值路徑與目標；推論積分器多次呼叫模型。單次 forward 不等於完整生成。','mdgen/transport/transport.py','class Transport:','temporal'),
        node('inference','選擇生成任務入口','sim / TPS / design / upsampling','條件構形 → 生成軌跡／序列','四種入口分別設定條件；圖中以 sim 的 wrapper.inference 呼叫為代表，後續各章分別展開。','sim_inference.py','atom14, _ = model.inference(expanded_batch)','condition'),
        node('export','重建與輸出原子軌跡','geometry / protein / utils','Atom14 → PDB / XTC','把生成 latent 還原到構形，並依設定處理輸出幾何保護；最後寫出可分析的軌跡。','sim_inference.py',"traj.save(os.path.join(args.out_dir",'geometry'),
        node('analysis','比較構形與動力學統計','scripts/analyze_* / mdgen.analysis','features / TICA / MSM / reports','分析腳本讀生成與參考資料，檢查幾何、狀態分布或轉移行為；不是另一個神經層。','scripts/analyze_peptide_sim.py','def main(name):','norm'),
        node('scale','獨立驗證全長工程路徑','scale_validate.py / full_length','確定性 fixture → 工程報告','局部 attention、全域摘要、Hyena、重建與測試組成規模驗證。它不代表已訓練的物理 MD 模型。','mdgen/full_length/scale_validation.py','def main(argv:','geometry'),
    ],[edge('prep','dataset'),edge('dataset','train'),edge('train','wrapper','訓練呼叫','condition'),edge('wrapper','model'),edge('wrapper','transport','訓練／取樣','condition'),edge('transport','model','多次模型評估','condition'),edge('inference','wrapper','推論條件','condition'),edge('wrapper','export','生成後重建'),edge('export','analysis')],
    ['這是跨模組關係圖，訓練與推論是不同執行模式，不是同一次依序全部執行。','job/ 管理環境與叢集提交；tests/、docs/ 與 splits/ 的完整內容由後續章節和全檔案索引涵蓋。'])]
    for chart in old['charts']:
        chart.pop('svg', None)
        chart['summary'] = chart.pop('subtitle')
        if chart['id']=='02-latent-layer':
            chart['summary']='依功能逐步展開空間、時間與 MLP 子層；條件與殘差保留為獨立支路。'
        chart['kind'] = 'neural'
        chart['steps'] = [dict(node=n['id'], explanation=n['description']) for n in chart['nodes']]
        for n in chart['nodes']:
            n.pop('x', None)
            n.pop('y', None)
        for e in chart['edges']:
            e.pop('points', None)
        chapters.append(chart)
    lm, mha, layers = 'mdgen/model/latent_model.py', 'mdgen/model/mha.py', 'mdgen/model/layers.py'
    chapters.append(chapter('neural-rope', '注意力內部：Q、K、V 與 RoPE',
        '以啟用 RoPE 的明確運算路徑解釋注意力；S 可以是殘基 L 或影格 T。', [
        node('x','先把序列維移到前面','AttentionWithRoPE','[M,S,C] → [S,M,C]','空間時 M=B×T、S=L；時間時 M=B×L、S=T。這是重排，不是新增資料。',lm,'x = x.transpose(0, 1)','input'),
        node('qkv','產生查詢、鍵與值','q_proj / k_proj / v_proj','各自 C → C','同一輸入經三個獨立投影；Q 問要找誰，K 提供配對特徵，V 攜帶要傳的內容。',mha,'q = self.q_proj(query)'),
        node('heads','拆成多個注意力頭','reshape + scaling','[M×H,S,C/H]','Q 乘 head_dim 的負二分之一次方；每個 head 處理一部分通道。',mha,'q *= self.scaling','norm'),
        node('rope','讓 Q 與 K 帶有位置資訊','RotaryEmbedding','Q、K 形狀不變','RoPE 旋轉 Q/K；不在此旋轉 V。關閉 RoPE 時可走 PyTorch 快速路徑。',mha,'q, k = self.rot_emb(q, k)','condition',True),
        node('scores','比較每個查詢與所有鍵','Q × K transpose','[M×H,S,S]','矩陣每列對應一個 query；列內是各 key 的相似分數。',mha,'attn_weights = torch.bmm(q, k.transpose(1, 2))','spatial'),
        node('mask','排除不可看的位置','padding / attention mask','[M×H,S,S]','padding 對應分數設為負無限；mask 是可見性規則，不是另一層資料。',mha,'attn_weights = attn_weights.masked_fill(','condition'),
        node('prob','把分數轉成混合比例','softmax + dropout','[M×H,S,S]','沿 key 維做 softmax；訓練時可再套用 dropout。圖上光點不代表真實注意力權重。',mha,'attn_weights_float = utils_softmax','spatial'),
        node('mix','用權重混合 V 的內容','attention probabilities × V','[M×H,S,C/H]','每個 query 收集所有可見 key 對應的 V，再依權重相加。',mha,'attn = torch.bmm(attn_probs, v)','spatial'),
        node('out','合併注意力頭再輸出','out_proj + transpose','[M,S,C]','合併 H 個 head，做 C→C 投影，再由 AttentionWithRoPE 還原 batch-first。殘差由外層負責。',mha,'attn = self.out_proj(attn)','linear'),
    ], [edge('x','qkv'),edge('qkv','heads'),edge('heads','rope'),edge('rope','scores'),edge('scores','mask'),edge('mask','prob'),edge('prob','mix'),edge('qkv','mix','V 支路','condition'),edge('mix','out')],
        ['這是 RoPE 明確路徑；本檔亦包含非 RoPE 快速路徑、增量快取與 encoder-decoder 通用功能，不代表 MDGen 每次都啟用。']))
    chapters.append(chapter('neural-time-head', '生成時間條件與最終輸出頭',
        '生成進度 t 控制特徵如何調變；它不是軌跡的影格索引 T。', [
        node('t','輸入生成進度','t × time_multiplier','[B]','模型把流匹配時間 t 乘上 time_multiplier；預設值由解析器決定。',lm,'t = self.t_embedder','input'),
        node('freq','展開不同頻率','cos / sin','[B,256]','先算 t×frequency，再串接 cos 與 sin，將標量換成頻率特徵。',layers,'embedding = torch.cat([torch.cos(args), torch.sin(args)]','temporal'),
        node('mlp','學習時間的隱藏表示','Linear → SiLU → Linear','256 → C → C','TimestepEmbedder 的兩層 MLP 把頻率特徵轉成條件向量。',layers,'t_emb = self.mlp(t_freq)'),
        node('main','主幹取得三組閘門參數','SiLU → Linear(C,9C)','三組 shift、scale、gate','空間、時間與 MLP 三個子層各有自己的調變參數。',lm,'nn.Linear(self.embed_dim, 9 * self.embed_dim','condition'),
        node('features','最後一層主幹特徵','LatentMDGenLayer output','[B,T,L,C]','主幹的最後輸出進入 FinalLayer；尚不是座標重建結果。',lm,'latent = self.emb_to_latent','input'),
        node('headcond','輸出頭只取平移與縮放','SiLU → Linear(C,2C)','shift、scale','FinalLayer 的條件投影只輸出兩組參數，沒有 gated residual。',layers,'shift, scale = self.adaLN_modulation(c).chunk','condition'),
        node('norm','標準化後套用時間條件','LayerNorm + modulate','[B,T,L,C]','先沿 C 正規化，再 x×(1+scale)+shift；不混合殘基或時間。',layers,'x = modulate(self.norm_final(x), shift, scale)','norm'),
        node('linear','投影回潛在維度','Linear(C,D)','[B,T,L,D]','此處輸出取決於 transport 的 prediction 類型。取樣器需反覆呼叫，才形成生成結果。',layers,'x = self.linear(x)','linear'),
    ], [edge('t','freq'),edge('freq','mlp'),edge('mlp','main','主幹調變','condition'),edge('mlp','headcond','輸出頭調變','condition'),edge('features','norm'),edge('headcond','norm','shift / scale','condition'),edge('norm','linear')]))
    p, a = 'mdgen/full_length/partition.py', 'mdgen/full_length/adaptive.py'
    chapters.append(chapter('neural-partition', '局部分區：core、ghost 與唯一擁有者',
        '按起始 Cα 座標建立固定鄰域；ghost 只提供上下文，core 負責回寫。', [
        node('coords','讀取有效殘基與座標','coordinates / mask / chain / index','[L,3] 與 [L]','空間位置、鏈別、序列編號一起決定哪些殘基互為鄰居。',p,'def build_spatial_partition','input'),
        node('cells','把位置分配到空間格子','floor(Cα / cell_width)','整數格子座標','相同格子的有效殘基先分在一起；這不是隨機分組。',p,'cell_coords = torch.floor','geometry'),
        node('core','限制每組核心數量','core_capacity','每組 ≤ core_capacity','大格子按 core_capacity 分批，每個有效殘基只屬於一個 core 群組。',p,'core_list = cell_members[start : start + core_capacity]','geometry'),
        node('covalent','優先保留同鏈序列鄰居','covalent_span','依鏈別與殘基編號','先用序列距離收集共價鄰居，避免單純空間分組切斷近鄰上下文。',p,'for delta in range(1, covalent_span + 1):','condition'),
        node('near','再補充空間鄰居','contact_radius','候選 ghost','用到 core 的最小距離篩選空間候選，排除 core 本身。',p,'distances = torch.cdist','geometry'),
        node('ghost','按優先順序截取 ghost','ghost_capacity','每組 ≤ ghost_capacity','共價鄰居優先、其餘按距離排序；ghost 可被不同群組重複讀取。',p,'ghost_list = (priority +','geometry'),
        node('owner','保存唯一擁有關係','owner_group','[L]','owner_group 指定每個有效殘基的 core 群組，供 scatter_core 只回寫一次。',p,'owner_group[core] = group_index','condition'),
        node('validate','檢查無重複、無漏掉','validate_ownership','OwnershipValidation','分區建完必須驗證擁有關係；無效會拋出錯誤。',p,'validation = partition.validate_ownership()','norm'),
        node('adaptive','可選：到視窗邊界才重建','AdaptiveSpatialPartitioner','位移閾值 + window_boundary','獨立 adaptive 工具：必須到視窗邊界且最大位移達閾值才重新分區。主模型 forward 不會每層自動呼叫它。',a,'if not window_boundary or trigger < self.displacement_threshold:','condition',True),
    ], [edge('coords','cells'),edge('cells','core'),edge('core','covalent'),edge('core','near'),edge('covalent','ghost','優先','condition'),edge('near','ghost'),edge('core','owner'),edge('ghost','validate'),edge('owner','validate'),edge('coords','adaptive','獨立可選工具','optional'),edge('adaptive','cells','重建','optional')],
        ['圖中 AdaptiveSpatialPartitioner 是另一路工具，不表示主模型預設使用自適應分區。']))
    return chapters


def git_paths():
    output = subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT)
    candidates = sorted(set(output.decode().split('\0')) - {''})
    included, excluded = [], []
    for path in candidates:
        if path.startswith(EXCLUDED) or '/__pycache__/' in path or path == 'mdgen/MERMAID.md':
            excluded.append(path)
        elif (ROOT / path).is_file():
            included.append(path)
    return included, excluded


def category(path):
    if path.startswith('mdgen/model/'): return '神經模型'
    if path.startswith('mdgen/full_length/'): return '全長模型與規模驗證'
    if path.startswith('mdgen/transport/'): return '流匹配與取樣'
    if path.startswith('mdgen/'): return '資料、幾何與共用工具'
    if path.startswith('tests/'): return '測試'
    if path.startswith('scripts/'): return '資料準備與分析腳本'
    if path.startswith('job/'): return '叢集工作提交'
    if path.startswith('splits/'): return '資料切分清單'
    if path.endswith('.py'): return '執行入口'
    return '文件與專案資產'


def inventory(paths, chapters):
    known = set(paths)
    result = []
    for path in paths:
        raw = (ROOT / path).read_bytes()
        text = raw.decode('utf-8',errors='replace')
        item = dict(path=path,category=category(path),sha256=hashlib.sha256(raw).hexdigest(),
                    summary='',symbols=[],imports=[],calls=[],chapters=[],bytes=len(raw))
        mentions = [n['description'] for c in chapters for n in c['nodes'] if n['source']['file']==path]
        if path.endswith('.py'):
            tree = ast.parse(text,filename=path)
            doc = ast.get_docstring(tree)
            item['summary'] = (doc.strip().splitlines()[0] if doc else '')
            def add_symbol(n, parent='', parent_is_class=False):
                fullname = f'{parent}.{n.name}' if parent else n.name
                kind = '類別' if isinstance(n,ast.ClassDef) else '方法' if parent_is_class else '巢狀函式' if parent else '函式'
                doc = ast.get_docstring(n) or ''
                signature = ast.unparse(n.args) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) else ', '.join(ast.unparse(b) for b in n.bases)
                item['symbols'].append(dict(name=fullname,kind=kind,line=n.lineno,end_line=n.end_lineno,
                    signature=signature,description=doc.strip(),parent=parent))
                visit_definitions(n,fullname,isinstance(n,ast.ClassDef))
            def visit_definitions(n,parent='',parent_is_class=False):
                for child in ast.iter_child_nodes(n):
                    if isinstance(child,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                        add_symbol(child,parent,parent_is_class)
                    else:visit_definitions(child,parent,parent_is_class)
            visit_definitions(tree)
            for n in ast.walk(tree):
                modules=[]
                if isinstance(n,ast.Import): modules=[a.name for a in n.names]
                elif isinstance(n,ast.ImportFrom):
                    base=path.removesuffix('.py').split('/')[:-1]
                    if n.level:
                        prefix=base[:len(base)-n.level+1]
                        module='.'.join(prefix+([n.module] if n.module else []))
                    else: module=n.module or ''
                    modules=[module]+[f'{module}.{a.name}'.strip('.') for a in n.names]
                for module in modules:
                    stem=module.replace('.','/')
                    match=next((p for p in [stem+'.py',stem+'/__init__.py'] if p in known),None)
                    if match and match != path: item['imports'].append(match)
            # Only statically obvious same-module/self calls; not a dynamic trace.
            symbol_names={s['name'] for s in item['symbols']}
            for top in tree.body:
                owners=[(top.name,top)] if isinstance(top,(ast.FunctionDef,ast.AsyncFunctionDef)) else [(f'{top.name}.{n.name}',n) for n in top.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))] if isinstance(top,ast.ClassDef) else []
                for owner, function in owners:
                    for call in ast.walk(function):
                        if not isinstance(call,ast.Call): continue
                        target=call.func.id if isinstance(call.func,ast.Name) else None
                        if isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id in ('self','cls') and '.' in owner:
                            target=owner.rsplit('.',1)[0]+'.'+call.func.attr
                        if target in symbol_names: item['calls'].append(dict(source=owner,target=target,line=call.lineno))
        elif path.endswith('.csv'):
            rows=list(csv.reader(text.splitlines()))
            item['summary']=f'資料切分清單：{max(0,len(rows)-1)} 筆記錄；欄位：'+', '.join(rows[0] if rows else [])
            item['columns']=rows[0] if rows else []
            item['row_count']=max(0,len(rows)-1)
        elif path.endswith('.sh'):
            for i,line in enumerate(text.splitlines(),1):
                match=re.match(r'\s*(?:function\s+)?([\w]+)\s*\(\)\s*\{',line)
                if match: item['symbols'].append(dict(name=match[1],kind='Shell 函式',line=i,description='',signature='',parent=''))
                for token in re.findall(r'[\w./-]+\.py',line):
                    if token in known: item['imports'].append(token)
            item['summary']='Shell 工作流程；參數、環境與實際啟動入口見原始碼。'
        elif path.endswith('.md'):
            headings=[(i,l.lstrip('# ').strip()) for i,l in enumerate(text.splitlines(),1) if l.startswith('#')]
            item['summary']=headings[0][1] if headings else '專案說明文件'
            item['symbols']=[dict(name=h,kind='文件章節',line=i,description='文件標題；不是可執行 layer。',signature='',parent='') for i,h in headings]
        elif path.endswith('.png'): item['summary']='專案既有圖像資產；不屬於神經模型的計算層。'
        elif path == 'LICENSE': item['summary']='授權條款；不是執行流程。'
        elif path == '.gitignore': item['summary']='Git 忽略規則；決定哪些本機產物不納入版本控制。'
        if mentions: item['summary']='；'.join(mentions[:2])
        if not item['summary']: item['summary']=f"{item['category']}：提供本檔定義的元件，詳見符號與來源位置。"
        item['imports']=sorted(set(item['imports']))
        item['chapters']=[c['id'] for c in chapters if any(n['source']['file']==path for n in c['nodes'])]
        result.append(item)
    return result


def file_pages(files):
    pages=[]
    for number,item in enumerate(files,1):
        symbols=item['symbols'] or [dict(name=item['path'],kind=item['category'],line=1,description=item['summary'],signature='')]
        item['pages']=[]
        for start in range(0,len(symbols),8):
            id=f'file-{number:03d}-{start//8+1:02d}'
            item['pages'].append(id)
            nodes=[]
            for j,s in enumerate(symbols[start:start+8]):
                lines=(ROOT/item['path']).read_text(errors='replace').splitlines() if not item['path'].endswith('.png') else []
                line=s['line']
                anchor=lines[line-1].strip() if lines and line<=len(lines) else ''
                nodes.append(dict(id=f's{j}',label=s['name'],zh=s['kind']+'：'+s['name'],shape=s.get('signature') or '結構索引，非執行張量',
                    description=s.get('description') or f"{s['kind']} {s['name']} 定義於第 {line} 行；用途需結合呼叫端理解。",source=dict(file=item['path'],line=line,anchor=anchor),kind='norm',optional=False))
            name_to_id={s['label']:s['id'] for s in nodes}
            edges=[edge(name_to_id[e['source']],name_to_id[e['target']],'靜態呼叫候選','condition') for e in item['calls'] if e['source'] in name_to_id and e['target'] in name_to_id]
            page=chapter(id,f"{item['path']} · {start//8+1}",item['summary'],nodes,edges,
                ['此頁為原始碼結構，不代表神經層或實際執行順序。','僅顯示同頁可靜態解析的本檔呼叫；動態派發與跨頁關係未畫成箭頭。'])
            page['kind']='structure'
            pages.append(page)
    return pages


def validate(chapters):
    ids=set()
    for c in chapters:
        if c['id'] in ids: raise ValueError('Duplicate chapter '+c['id'])
        ids.add(c['id'])
        nodes={n['id']:n for n in c['nodes']}
        if len(nodes)!=len(c['nodes']): raise ValueError('Duplicate node '+c['id'])
        for n in c['nodes']:
            ref=n['source']
            if ref['anchor']:
                lines=(ROOT/ref['file']).read_text().splitlines()
                if ref['anchor'] not in lines[ref['line']-1]:
                    raise ValueError(f"Source mismatch {c['id']} {n['id']} {ref}")
            if not re.search('[\u3400-\u9fff]',n['zh']): raise ValueError('Missing Chinese '+n['id'])
        for e in c['edges']:
            if e['source'] not in nodes or e['target'] not in nodes: raise ValueError(f'Invalid edge {c["id"]} {e}')
        if {s['node'] for s in c['steps']} != set(nodes): raise ValueError('Missing steps '+c['id'])


def build():
    chapters=neural_chapters()
    for filename in ('core_chapters.json','workflow_chapters.json'):
        value=json.loads((HERE/filename).read_text())
        chapters.extend(value if isinstance(value,list) else value['chapters'])
    paths,excluded=git_paths()
    files=inventory(paths,chapters)
    pages=file_pages(files)
    validate(chapters+pages)
    for c in chapters+pages:
        c['poster']=f'posters/{c["id"]}.png'
        if c['kind']!='structure': c['video']=f'videos/{c["id"]}.mp4'
    coverage=dict(files=len(files),python_files=sum(f['path'].endswith('.py') for f in files),
        symbols=sum(len(f['symbols']) for f in files),chapters=len(chapters),file_pages=len(pages),
        source_anchors=sum(len(c['nodes']) for c in chapters),excluded=excluded,
        exclusions='排除 .git、忽略的資料/權重/快取、.omx、.scratch 與本次/先前生成的視覺化產物；納入版本管理檔案及未忽略的人工原始碼/文件。',
        evidence='AST 結構索引 + 人工核對流程。不是模型執行追蹤，沒有載入 checkpoint 或量測 activation。')
    data=dict(title='MDGen 全 repo：ManimML 動畫與逐步教學',built_at=datetime.now(timezone.utc).isoformat(),
        scope=coverage['exclusions'],chapters=chapters,files=files,file_pages=pages,coverage=coverage)
    (HERE/'tour-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (HERE/'coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
    template=HERE/'tour.html'
    if template.exists():
        encoded=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
        (HERE/'index.html').write_text(template.read_text().replace('__TOUR_DATA__',encoded,1))
    print(json.dumps({k:v for k,v in coverage.items() if k not in ('excluded',)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    build()
