"""Inspect MP4 movie/track edit-list timing without decoding or modifying media."""
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def inspect(path):
    b=path.read_bytes()
    tracks=[]
    movie={}
    def scan(start,end,track=None):
        while start+8<=end:
            size,kind=struct.unpack('>I4s',b[start:start+8]); header=8
            if size==1:
                size=struct.unpack('>Q',b[start+8:start+16])[0];header=16
            if size==0:
                size=end-start
            if size<header:
                raise ValueError('Invalid MP4 box size')
            pos=start+header
            if kind==b'trak':
                tr={};tracks.append(tr);scan(pos,start+size,tr)
            elif kind in (b'moov',b'mdia',b'edts'):
                scan(pos,start+size,track)
            elif kind in (b'mvhd',b'mdhd'):
                version=b[pos];offset=pos+(20 if version else 12)
                scale=struct.unpack('>I',b[offset:offset+4])[0]
                duration=struct.unpack('>Q' if version else '>I',b[offset+4:offset+(12 if version else 8)])[0]
                target=movie if kind==b'mvhd' else track
                target.update(timescale=scale,duration=duration)
            elif kind==b'hdlr':
                track['handler']=b[pos+8:pos+12].decode()
            elif kind==b'elst':
                version=b[pos];count=struct.unpack('>I',b[pos+4:pos+8])[0]
                offset=pos+8;entries=[]
                for _ in range(count):
                    duration,media_time=struct.unpack('>Qq' if version else '>Ii',b[offset:offset+(16 if version else 8)])
                    offset+=20 if version else 12
                    entries.append([duration,media_time])
                track['edit_entries']=entries
            start+=size
    scan(0,len(b))
    for t in tracks:
        t['presentation_duration']=sum(v[0] for v in t.get('edit_entries',[]))/movie['timescale']
        t['media_offsets_seconds']=[v[1]/t['timescale'] for v in t.get('edit_entries',[])]
    return movie['duration']/movie['timescale'],tracks

if __name__=='__main__':
    source=json.loads((Path(__file__).parent/'evidence.json').read_text(encoding='utf-8'))
    rows=[]
    for r in source['problem1']['movie_durations']:
        video,clip=r['id'].split('$_$')
        duration,tracks=inspect(ROOT/'DATA/attachment_1_raw_samples/videos'/video/(clip+'.mp4'))
        rows.append({**r,'container_seconds':duration,'tracks':tracks})
    (Path(__file__).parent/'media_timeline_evidence.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved timing for',len(rows),'videos')
