#!/usr/bin/env python3
"""Render actual ManimML custom layers plus source-grounded Manim flow scenes.

No torch/model imports, weights or simulated activations. ManimML implements
the custom layer animation interface; connections come from the audited graph,
not a fabricated fully-connected sequential model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import numpy as np
from manim import (
    AnimationGroup, Arrow, Circle, Create, DashedLine, Dot, FadeIn, FadeOut,
    Group, Indicate, Line, MoveAlongPath, Rectangle, RoundedRectangle, Scene,
    Succession, Text, VGroup, VMobject, config, linear, tempconfig,
)
from manim_ml.neural_network.layers.parent_layers import NeuralNetworkLayer

HERE = Path(__file__).resolve().parent
FONT = 'Noto Sans CJK TC'
BG, WHITE, MUTED, TEAL = '#101b2d', '#f2f6ff', '#a8bad0', '#57dbc5'
COLORS = {'input':'#66cda0','linear':'#f6ba77','spatial':'#c4a1ef','temporal':'#edcc70',
          'geometry':'#8dbbff','norm':'#b0bdd2','condition':'#57dbc5','add':'#f39191'}
EDGE_COLORS = {'data':'#a8bad0','condition':TEAL,'optional':'#c4a1ef','residual':'#f39191'}


def wrap(value, columns=42, max_lines=None):
    """Wrap by display width, retaining all content unless explicitly previewing."""
    import unicodedata
    lines=[]
    for paragraph in str(value).splitlines() or ['']:
        line=''; width=0
        for char in paragraph:
            size=2 if unicodedata.east_asian_width(char) in 'WF' else 1
            if width+size>columns and line:
                lines.append(line); line=''; width=0
            line+=char; width+=size
        lines.append(line)
    if max_lines and len(lines)>max_lines:
        lines=lines[:max_lines]; lines[-1]=lines[-1].rstrip()+'…'
    return '\n'.join(lines)


def text(value, size=24, color=WHITE, width=None):
    t=Text(str(value),font=FONT,font_size=size,color=color,line_spacing=.9,disable_ligatures=True)
    if width and t.width>width: t.scale_to_fit_width(width)
    return t


class RepoLayer(NeuralNetworkLayer):
    """ManimML extension: an honest labeled module, not invented neurons."""
    def __init__(self, spec, width=5.9, height=1.75, compact=False):
        super().__init__()
        self.spec=spec
        self.box=RoundedRectangle(width=width,height=height,corner_radius=.13,
            stroke_color=COLORS.get(spec['kind'],MUTED),stroke_width=2.5,
            fill_color='#1d2d45',fill_opacity=1)
        title=text(wrap(spec['zh'],28 if compact else 43,2),17 if compact else 27,width=width-.32)
        label=text(wrap(spec['label'],32 if compact else 55,1),12 if compact else 16,color=MUTED,width=width-.32)
        content=VGroup(title,label).arrange(np.array([0,-1,0]),buff=.10)
        if not compact:
            shape=text(wrap(spec['shape'],60,2),18,color=TEAL,width=width-.35)
            content.add(shape); content.arrange(np.array([0,-1,0]),buff=.12)
        if content.height>height-.25: content.scale_to_fit_height(height-.25)
        self.add(self.box,content)

    def construct_layer(self,input_layer=None,output_layer=None,**kwargs):
        # Geometry is explicit to preserve branches/residuals in the source graph.
        return self

    def make_forward_pass_animation(self,layer_args=None,**kwargs):
        return Indicate(self.box,color=TEAL,scale_factor=1.035,run_time=kwargs.get('run_time',.6))


class ChapterScene(Scene):
    def __init__(self,chapter,poster=False,hold=4.5,**kwargs):
        self.chapter=chapter; self.poster=poster; self.hold=hold; self.timings=[]
        super().__init__(**kwargs)

    def header(self):
        title=text(self.chapter['title'],32,width=14.6).move_to([0,3.8,0])
        subtitle=text(wrap(self.chapter['summary'],112,2),19,color=MUTED,width=14.6).move_to([0,3.13,0])
        self.add(title,subtitle)
        footer='ManimML 自訂 Layer × Manim ｜ 繁體中文教學 ｜ 資料流示意，非 activation 測量'
        if self.chapter['kind']=='structure': footer='原始碼結構索引 ｜ 定義與靜態呼叫候選，不是神經層或執行順序'
        self.add(text(footer,14,color=MUTED,width=14.5).move_to([0,-4.12,0]))

    def construct(self):
        self.header()
        if self.poster:
            self.overview(); return
        nodes={n['id']:n for n in self.chapter['nodes']}
        steps=self.chapter['steps']
        # The numbered ribbon is navigation only, never a synthetic data edge.
        ribbon=VGroup(*[text(str(i+1),16,color=MUTED) for i in range(len(steps))]).arrange(np.array([1,0,0]),buff=.30)
        if ribbon.width>13.5: ribbon.scale_to_fit_width(13.5)
        ribbon.move_to([0,2.5,0]); self.add(ribbon)
        previous=Group()
        for index, step in enumerate(steps):
            start_time=self.time
            n=nodes[step['node']]
            ribbon.set_color(MUTED); ribbon[index].set_color(TEAL)
            current=RepoLayer(n).move_to([0,.78,0])
            incoming=[e for e in self.chapter['edges'] if e['target']==n['id']]
            outgoing=[e for e in self.chapter['edges'] if e['source']==n['id']]
            panel=Group(current)
            paths=[]
            for side,edges in [(-1,incoming),(1,outgoing)]:
                visible=edges[:3]
                for j,e in enumerate(visible):
                    other=nodes[e['source'] if side<0 else e['target']]
                    y=.8+(len(visible)-1)*.60-j*1.20
                    box=RepoLayer(other,width=2.9,height=.93,compact=True).move_to([side*5.9,y,0])
                    color=EDGE_COLORS.get(e['kind'],MUTED)
                    start=box.get_right() if side<0 else current.get_right()
                    end=current.get_left() if side<0 else box.get_left()
                    arr=Arrow(start,end,buff=.10,color=color,stroke_width=2.3,max_tip_length_to_length_ratio=.12)
                    if e['kind']!='data':
                        line=DashedLine(start,end,color=color,stroke_width=1.5,dash_length=.07)
                        panel.add(line)
                    panel.add(arr,box)
                    if side<0: paths.append(arr)
                if len(edges)>3:
                    panel.add(text(f'另有 {len(edges)-3} 條連線，見 HTML 完整圖',12,color=MUTED,width=3).move_to([side*5.9,-1.5,0]))
            panel.add(text('流入：來源／條件',15,color=MUTED).move_to([-5.9,2.04,0]),
                      text('本步驟',16,color=TEAL).move_to([0,2.04,0]),
                      text('流出：後續／分支',15,color=MUTED).move_to([5.9,2.04,0]))
            explanation=step['explanation']
            # Entire narration is retained as subtitles; preview panel wraps to fit.
            panel.add(text(wrap(explanation,112),24,width=14.0).move_to([0,-2.06,0]))
            ref=n['source']
            panel.add(text(f"來源 {ref['file']}:{ref['line']}  ·  {index+1}/{len(steps)}",15,color=MUTED,width=14.2).move_to([0,-3.26,0]))
            edge_notes='  ／  '.join(('流入 ' if e in incoming else '流出 ')+(e.get('label') or e['kind']) for e in (incoming+outgoing))
            panel.add(text(wrap(edge_notes,120,2),14,color=TEAL,width=14).move_to([0,-3.64,0]))
            if len(previous): self.remove(previous)
            self.add(panel)
            if paths:
                dots=[Dot(path.get_start(),radius=.06,color=TEAL) for path in paths]
                for dot in dots:self.add(dot)
                self.play(AnimationGroup(*[MoveAlongPath(dot,path,rate_func=linear) for dot,path in zip(dots,paths)]),run_time=.65)
                for dot in dots:self.remove(dot)
            else: self.wait(.65)
            self.play(current.make_forward_pass_animation(run_time=.55))
            self.wait(self.hold)
            self.timings.append((start_time,self.time))
            previous=panel

    def overview(self):
        specs=self.chapter['nodes']; count=len(specs)
        cols=4 if count>9 else 3
        rows=math.ceil(count/cols)
        width=3.25 if cols==4 else 4.25
        height=min(1.28,4.8/max(rows,1))
        positions={}
        for i,n in enumerate(specs):
            x=(i%cols-(cols-1)/2)*(15.0/cols)
            y=1.93-(i//cols)*(5.0/max(rows,1))
            positions[n['id']]=(x,y)
        # Source relationships are drawn lightly; page is an overview, not a trace.
        for e in self.chapter['edges']:
            if e['source']==e['target']: continue
            a=np.array([*positions[e['source']],0]); b=np.array([*positions[e['target']],0])
            if np.linalg.norm(a-b)<.01:continue
            arr=Arrow(a,b,buff=.8,color=EDGE_COLORS.get(e['kind'],MUTED),stroke_width=1.3,max_tip_length_to_length_ratio=.06)
            arr.set_opacity(.4);self.add(arr)
        for i,n in enumerate(specs):
            card=RepoLayer(n,width=width,height=height,compact=True)
            card.move_to([*positions[n['id']],0]);self.add(card)
            self.add(text(str(i+1),13,color=TEAL).next_to(card,np.array([-1,0,0]),buff=.07))
        note='逐步影片會放大每一層、顯示輸入輸出形狀與來源；完整分支與說明另見 HTML。'
        if self.chapter['kind']=='structure':note='每個符號的完整簽名、原文註解、來源與依賴，請在 HTML「全檔案索引」查看。'
        self.add(text(note,17,color=MUTED,width=14.5).move_to([0,-3.53,0]))


def srt(chapter, hold, timings=None):
    def stamp(seconds):
        ms=round(seconds*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
    lines=[]
    for i,step in enumerate(chapter['steps']):
        n=next(n for n in chapter['nodes'] if n['id']==step['node'])
        begin,end=timings[i] if timings else (i*(hold+1.2),(i+1)*(hold+1.2))
        lines.extend([str(i+1),f'{stamp(begin)} --> {stamp(end)}',
                      n['zh']+'\n'+step['explanation'],''])
    return '\n'.join(lines)


def render_one(chapter, poster=False, hold=4.5):
    destination=HERE/('posters' if poster else 'videos')
    destination.mkdir(exist_ok=True)
    output=destination/(chapter['id']+('.png' if poster else '.mp4'))
    with tempconfig(dict(pixel_width=1280,pixel_height=720,frame_width=16,frame_height=9,
        frame_rate=15,background_color=BG,preview=False,write_to_movie=not poster,
        save_last_frame=poster,disable_caching=True,verbosity='ERROR',progress_bar='none',
        media_dir=str(HERE/'render-cache'/('posters' if poster else 'videos')),output_file=chapter['id'])):
        scene=ChapterScene(chapter,poster=poster,hold=hold)
        scene.render()
        generated=scene.renderer.file_writer.image_file_path if poster else scene.renderer.file_writer.movie_file_path
        shutil.copyfile(generated,output)
    if not poster:
        (destination/(chapter['id']+'.srt')).write_text(srt(chapter,hold,scene.timings))
    return dict(id=chapter['id'],file=str(output.relative_to(HERE)),bytes=output.stat().st_size,
                mode='poster' if poster else 'video',manimml_custom_layer=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--chapter')
    parser.add_argument('--posters',action='store_true')
    parser.add_argument('--files-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--hold',type=float,default=4.5)
    args=parser.parse_args()
    data=json.loads((HERE/'tour-data.json').read_text())
    charts=data['file_pages'] if args.files_only else data['chapters']+(data['file_pages'] if args.posters else [])
    if args.chapter: charts=[c for c in charts if c['id']==args.chapter]
    if not charts:raise ValueError('No matching chapters')
    report=[]
    for c in charts:
        poster=args.posters or args.files_only
        expected=HERE/('posters' if poster else 'videos')/(c['id']+('.png' if poster else '.mp4'))
        if args.resume and expected.exists() and expected.stat().st_size>1000:
            print('Existing',expected.name,flush=True);continue
        item=render_one(c,poster=poster,hold=args.hold); report.append(item)
        print(json.dumps(item,ensure_ascii=False),flush=True)
    name='posters-render.json' if args.posters or args.files_only else 'videos-render.json'
    (HERE/name).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
