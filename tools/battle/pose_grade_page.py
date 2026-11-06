#!/usr/bin/env python3
"""Local grading page for the Phase 7d frames (ours). Standard library only; binds to 127.0.0.1.

Frames come from raw videos, so they stay on this machine: nothing is uploaded and the folder is not committed.
Grades go to DIR/grades_<grader>.json after every answer.

    python3 tools/battle/pose_grade_page.py DIR [--port 8765] [--grader hyundo]    # then open http://127.0.0.1:8765
    python3 tools/battle/pose_grade_page.py DIR --severity GRADES.json     # only frames graded ok_bad: how bad is the skeleton
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = r"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>포즈 판정</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f6f4;--fg:#1c1c1a;--mut:#6b6b66;--card:#fff;--line:#d9d9d4;--on:#1f6f54;--onfg:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecece8;--mut:#a0a09a;--card:#242422;--line:#3c3c38;--on:#3aa582;--onfg:#0d1a15}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,'Noto Sans KR',sans-serif}
main{max-width:1240px;margin:auto;padding:12px 16px 24px}
header{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}h1{font-size:17px;margin:0}
.mut{color:var(--mut);font-size:13px}.bar{height:5px;background:var(--line);border-radius:3px;margin:8px 0 10px}.bar i{display:block;height:100%;background:var(--on);border-radius:3px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:8px}.two img{width:100%;display:block;border-radius:4px;background:#000;aspect-ratio:16/9}
.strip{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-top:6px;align-items:center}.strip img{width:100%;display:block;border-radius:3px;background:#000;aspect-ratio:16/9}
.cap{font-size:12px;color:var(--mut);margin:3px 0}.strip div{text-align:center;font-size:12px;color:var(--mut)}
.q{margin-top:12px}.q b{display:block;margin-bottom:5px}.opts{display:flex;flex-wrap:wrap;gap:6px}
button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:7px 11px;cursor:pointer}
button.on{background:var(--on);color:var(--onfg);border-color:var(--on)}kbd{font:12px ui-monospace,monospace;opacity:.65;margin-right:5px}
nav{display:flex;gap:8px;margin-top:14px;align-items:center}.note{width:100%;margin-top:8px;font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
@media (max-width:720px){.two{grid-template-columns:1fr}.strip{grid-template-columns:repeat(2,1fr)}}
</style><main>
<header><h1>포즈 판정</h1><span class="mut" id="pos"></span><span class="mut" id="saved"></span></header>
<div class="bar"><i id="bar"></i></div>
<div class="two"><div><img id="raw" alt="원본"><div class="cap">원본</div></div>
<div><img id="pose" alt="포즈"><div class="cap" id="capp"></div></div></div>
<div class="strip" id="strip"></div>
<div class="q"><b id="q1t"></b><div class="opts" id="q1"></div></div>
<div class="q" id="q2w"><b>이 순간의 동작 <span class="mut">(앞뒤 장면 참고)</span></b><div class="opts" id="q2"></div></div>
<input class="note" id="note" placeholder="메모 (선택)">
<nav><button id="prev">이전 (←)</button><button id="next">다음 (→)</button><span class="mut" id="done"></span></nav>
</main><script>
const POSE=[['ok','맞는 댄서, 뼈대 양호'],['ok_bad','맞는 댄서, 뼈대 틀어짐'],['wrong','다른 사람'],['unclear','판단 불가']];
const NOPOSE=[['visible','댄서가 보이는데 놓침'],['absent','댄서가 안 보임 (클로즈업, 전광판 등)'],['unclear','판단 불가']];
const MOVE=[['toprock','탑락'],['footwork','풋워크'],['power','파워무브'],['freeze','프리즈'],['transition','고다운·전환'],['notdance','춤 아님 (걷기, 대기, 인사)'],['unknown','모르겠음']];
const SEV=[['minor','경미: 손목·발목 등 끝 관절만 어긋남, 몸 구조는 맞음'],['limb','부분 실패: 팔이나 다리 하나가 통째로 틀림, 또는 좌우 뒤바뀜'],['structure','구조 실패: 몸통 방향이 틀리거나 상하가 뒤집혀 자세를 알아볼 수 없음']];
const KEYS='qwertyu';let F=[],G={},i=0,SM=false;const full=g=>g&&g.who&&(SM||g.move);
const $=id=>document.getElementById(id);
async function save(){const r=await fetch('/save',{method:'POST',body:JSON.stringify(G)});$('saved').textContent=r.ok?'저장됨':'저장 실패';}
function opts(el,list,key,keys){el.textContent='';const f=F[i],g=G[f.k]||{};list.forEach(([v,t],n)=>{const b=document.createElement('button');
 const kb=document.createElement('kbd');kb.textContent=keys[n];b.append(kb,t);if(g[key]===v)b.className='on';b.onclick=()=>pick(key,v);el.append(b);});}
function pick(key,v){const f=F[i];G[f.k]=Object.assign(G[f.k]||{},{[key]:v});save();render();
 const g=G[f.k];if(full(g)&&i<F.length-1&&(SM||key==='move')){i++;render();}}
function render(){const f=F[i],p=String(f.k).padStart(2,'0');$('raw').src=`/img/${p}_raw.jpg`;$('pose').src=`/img/${p}_pose.jpg`;
 $('capp').textContent=f.has_pose?'초록 상자와 뼈대 = 시스템이 고른 사람, 흰 상자 = 검출된 다른 사람':'시스템이 이 순간 포즈를 쓰지 않음 (흰 상자 = 검출된 사람)';
 const s=$('strip');s.textContent='';[['c0','−1초'],['c1','−0.5초'],[null,''],['c2','+0.5초'],['c3','+1초']].forEach(([c,t])=>{const d=document.createElement('div');
  if(c){const im=document.createElement('img');im.src=`/img/${p}_${c}.jpg`;im.alt=t;d.append(im);d.append(t);}else d.textContent='◀ 앞뒤 장면 ▶';s.append(d);});
 $('q1t').textContent=SM?'뼈대가 얼마나 틀어졌나요?':f.has_pose?'시스템이 고른 사람이 차례 중인 댄서인가요?':'이 순간 차례 중인 댄서가 보이나요?';
 opts($('q1'),SM?SEV:f.has_pose?POSE:NOPOSE,'who','1234');$('q2w').hidden=SM;if(!SM)opts($('q2'),MOVE,'move',KEYS);
 $('note').value=(G[f.k]||{}).note||'';$('pos').textContent=`${i+1} / ${F.length}`;
 const n=F.filter(x=>full(G[x.k])).length;$('bar').style.width=(100*n/F.length)+'%';$('done').textContent=n===F.length?'모두 끝났습니다. 창을 닫으셔도 됩니다.':`완료 ${n}개`;}
$('prev').onclick=()=>{if(i>0){i--;render();}};$('next').onclick=()=>{if(i<F.length-1){i++;render();}};
$('note').onchange=e=>{const f=F[i];G[f.k]=Object.assign(G[f.k]||{},{note:e.target.value});save();};
addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const f=F[i],a=SM?SEV:(f.has_pose?POSE:NOPOSE);
 if(e.key==='ArrowLeft')$('prev').click();else if(e.key==='ArrowRight')$('next').click();
 else if('1234'.includes(e.key)&&a[+e.key-1])pick('who',a[+e.key-1][0]);else if(!SM){const n=KEYS.indexOf(e.key);if(n>=0)pick('move',MOVE[n][0]);}});
(async()=>{SM=(await(await fetch('/mode')).json()).severity;F=await(await fetch('/frames')).json();G=await(await fetch('/grades')).json();const u=F.findIndex(x=>!full(G[x.k]));i=u<0?0:u;render();})();
</script></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--grader', default='hyundo')
    p.add_argument('--severity', default=None)
    a = p.parse_args()
    root = os.path.abspath(a.dir)
    out = os.path.join(root, f'grades_{a.grader}{"_severity" if a.severity else ""}.json')
    frames = json.load(open(os.path.join(root, 'frames.json')))
    if a.severity:
        prev = json.load(open(a.severity))
        frames = [r for r in frames if prev.get(str(r['k']), {}).get('who') == 'ok_bad']

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def send(self, body, ctype='application/json', code=200):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                self.send(PAGE.encode(), 'text/html; charset=utf-8')
            elif self.path == '/frames':
                self.send(json.dumps(frames).encode())
            elif self.path == '/mode':
                self.send(json.dumps({'severity': bool(a.severity)}).encode())
            elif self.path == '/grades':
                self.send(open(out, 'rb').read() if os.path.exists(out) else b'{}')
            elif self.path.startswith('/img/'):
                f = os.path.basename(self.path)
                q = os.path.join(root, f)
                if f.endswith('.jpg') and os.path.exists(q):
                    self.send(open(q, 'rb').read(), 'image/jpeg')
                else:
                    self.send(b'', code=404)
            else:
                self.send(b'', code=404)

        def do_POST(self):
            if self.path != '/save':
                return self.send(b'', code=404)
            body = self.rfile.read(int(self.headers['Content-Length']))
            json.loads(body)
            with open(out + '.tmp', 'wb') as f:
                f.write(body)
            os.replace(out + '.tmp', out)
            self.send(b'{}')

    print(f'http://127.0.0.1:{a.port}  ->  {out}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), H).serve_forever()


if __name__ == '__main__':
    main()
