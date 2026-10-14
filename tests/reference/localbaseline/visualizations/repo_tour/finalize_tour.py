#!/usr/bin/env python3
"""Verify rendered assets, build one MP4/SRT and a complete PDF atlas."""
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import tempfile

from PIL import Image

HERE=Path(__file__).resolve().parent


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))


def seconds(stamp):
    h,m,s,ms=map(int,re.split('[:,]',stamp));return h*3600+m*60+s+ms/1000


def timestamp(value):
    ms=round(value*1000)
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'


def main():
    data=json.loads((HERE/'tour-data.json').read_text())
    photos=[]
    for c in data['chapters']+data['file_pages']:
        path=HERE/c['poster']
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            assert im.size==(1280,720),(path,im.size)
            photos.append(im.convert('RGB'))
    photos[0].save(HERE/'repo-atlas.pdf',save_all=True,append_images=photos[1:],resolution=110,
        title='MDGen 全 repo 架構與檔案索引',author='MDGen source-grounded visualization',quality=95)
    for im in photos:im.close()
    media=[]; offset=0.; subtitles=[]; subtitle_index=1; metadata=[';FFMETADATA1']
    for c in data['chapters']:
        path=HERE/c['video']; info=probe(path)
        stream=next(s for s in info['streams'] if s['codec_type']=='video')
        duration=float(info['format']['duration'])
        assert stream['width']==1280 and stream['height']==720
        assert duration>len(c['steps'])*4 and duration<len(c['steps'])*8,(c['id'],duration)
        assert stream['codec_name']=='h264'
        raw=path.with_suffix('.srt').read_text()
        blocks=re.split(r'\n\s*\n',raw.strip())
        assert len(blocks)==len(c['steps'])
        end=0.
        for block in blocks:
            lines=block.splitlines(); a,b=lines[1].split(' --> ')
            begin,end=seconds(a),seconds(b)
            assert 0<=begin<end<=duration+.1,(path,end,duration)
            subtitles.extend([str(subtitle_index),f'{timestamp(offset+begin)} --> {timestamp(offset+end)}',*lines[2:],''])
            subtitle_index+=1
        # Decode an actual intermediate frame, not just container metadata.
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(duration/2),'-i',str(path),
                        '-frames:v','1','-f','null','-'],check=True)
        metadata.extend(['[CHAPTER]','TIMEBASE=1/1000',f'START={round(offset*1000)}',
                         f'END={round((offset+duration)*1000)}','title='+c['title'].replace('=','\\=')])
        media.append(dict(id=c['id'],title=c['title'],duration=duration,start=offset,
            steps=len(c['steps']),width=1280,height=720,codec='h264',frame_decoded=True,subtitle_end=end))
        offset+=duration
    with tempfile.TemporaryDirectory(prefix='mdgen-tour-merge-') as tmp:
        concat=Path(tmp)/'concat.txt'; meta=Path(tmp)/'chapters.txt'
        concat.write_text('\n'.join("file '"+str((HERE/c['video']).resolve()).replace("'","'\\''")+"'" for c in data['chapters']))
        meta.write_text('\n'.join(metadata)+'\n')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(concat),
            '-i',str(meta),'-map_metadata','1','-c','copy','-movflags','+faststart',str(HERE/'repo-tour-complete.mp4')],check=True)
    (HERE/'repo-tour-complete.srt').write_text('\n'.join(subtitles))
    merged=probe(HERE/'repo-tour-complete.mp4')
    assert abs(float(merged['format']['duration'])-offset)<1
    report=dict(coverage=data['coverage'],chapters=media,images=len(data['chapters'])+len(data['file_pages']),
        total_duration_seconds=offset,video_files=len(media),subtitles=subtitle_index-1,
        versions={p:importlib.metadata.version(p) for p in ['manim','manim-ml','ManimPango','pycairo']},
        actual_manimml='RepoLayer extends NeuralNetworkLayer; each highlighted layer calls make_forward_pass_animation.',
        model_executed=False,weights_loaded=False,reference_docs=[
            'https://github.com/helblazer811/ManimML',
            'https://github.com/helblazer811/ManimML/blob/main/manim_ml/neural_network/layers/parent_layers.py',
            'https://docs.manim.community/en/stable/tutorials/output_and_config.html'])
    (HERE/'render-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['images','video_files','subtitles','total_duration_seconds','versions']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
