"""Midterm deck (MLAVL reproduction + 1st improvement), readable without the speaker: a message line under every
title, plain-word terms, charts and drawn diagrams instead of dense tables.

    python slides/midterm/build2.py [out.pptx] [ablation.json]
    default out: presentations/2025198001_정현도_중간발표.pptx ; ablation.json = {"swap": rho, ...} fills the last table
"""
import json
import os
import sys

_ARGS = [os.path.abspath(a) for a in sys.argv[1:]]   # before lib chdirs

from lib import *  # noqa: F401,F403  (template deck, exemplar shapes, helpers; chdir to slides/midterm)
from lib import prs, S, find, title, clear, textbox, panel, bullets, bar, foot, table, arrow, drop_unused_rels
from lib import NAVY, BLUE, RED, GRAY, LIGHT, GREEN, NS
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

OUT = _ARGS[0] if len(_ARGS) > 0 else None
ABL_PATH = _ARGS[1] if len(_ARGS) > 1 else None
OUT = OUT or os.path.abspath('../../presentations/2025198001_정현도_중간발표.pptx')
ABL = json.load(open(ABL_PATH)) if ABL_PATH and os.path.exists(ABL_PATH) else {}
REPRO, GRAYBAR, MID = 'A6B0BB', 'A6B0BB', '6F9FD8'
DARK = '1A1A1A'


def lead(s, text):
    textbox(s, 0.44, 0.98, 12.7, 0.5, text, size=20, color=NAVY, bold=True)


def box(s, x, y, w, h, lines, fill=LIGHT, line=None, size=14, color=NAVY, dash=False, align='ctr', radius=0.08):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = radius
    if fill:
        shp.fill.solid(); shp.fill.fore_color.rgb = RGBColor.from_string(fill)
    else:
        shp.fill.background()
    if line:
        shp.line.color.rgb = RGBColor.from_string(line); shp.line.width = Pt(1.75)
        if dash:
            shp.line.dash_style = 4  # dash
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ('margin_left', 'margin_right'):
        setattr(tf, m, Inches(0.08))
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    for k, ln in enumerate(lines):
        if isinstance(ln, str):
            ln = (ln, {})
        text, o = ln
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = {'ctr': PP_ALIGN.CENTER, 'l': PP_ALIGN.LEFT}[align]
        r = p.add_run(); r.text = text
        r.font.size = Pt(o.get('size', size + 2 if k == 0 else size))
        r.font.bold = o.get('b', k == 0)
        r.font.name = 'Calibri'
        r.font.color.rgb = RGBColor.from_string(o.get('c', color))
    return shp


def arr(s, x, y, w=0.4, h=0.32, d='r', color='9AA5B1'):
    shape = {'r': MSO_SHAPE.RIGHT_ARROW, 'd': MSO_SHAPE.DOWN_ARROW}[d]
    a = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid(); a.fill.fore_color.rgb = RGBColor.from_string(color)
    a.line.fill.background(); a.shadow.inherit = False
    return a


def chart(s, x, y, w, h, cats, series, ymin, ymax, ttl, unit=0.05, fmt='0.000', legend=True, lsz=11):
    cd = CategoryChartData(); cd.categories = cats
    for name, vals, _ in series:
        cd.add_series(name, vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.has_title = True
    ch.chart_title.text_frame.text = ttl
    r = ch.chart_title.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(14); r.font.bold = True
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(13)
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = ymin, ymax, unit
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor.from_string('E3E7EB')
    va.tick_labels.font.size = Pt(11); va.format.line.fill.background()
    va.tick_labels.number_format = '0.00'; va.tick_labels.number_format_is_linked = False
    ch.category_axis.tick_labels.font.size = Pt(13)
    plot = ch.plots[0]
    plot.gap_width, plot.overlap = 55, -8
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt; dl.number_format_is_linked = False
    dl.font.size = Pt(lsz); dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for ser, (_, _, col) in zip(plot.series, series):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = RGBColor.from_string(col)
    return ch


def videos(s, specs):
    """specs: [(clip name, x, y, w, h)] -> movies that start on slide entry and loop (no click needed)"""
    from lxml import etree
    sp = []
    for name, x, y, w, h in specs:
        mv = s.shapes.add_movie(f'assets/clips/{name}.mp4', Inches(x), Inches(y), Inches(w), Inches(h),
                                poster_frame_image=f'assets/clips/{name}.jpg', mime_type='video/mp4')
        sp.append(mv.shape_id)
    t = s._element.find('{%s}timing' % NS['p'])
    if t is not None:
        s._element.remove(t)
    ids = iter(range(1, 1000))
    nid = lambda: next(ids)
    root, main, par3, par4 = nid(), nid(), nid(), nid()
    calls = ''
    for k, spid in enumerate(sp):
        a, b = nid(), nid()
        calls += (f'<p:par><p:cTn id="{a}" presetID="1" presetClass="mediacall" presetSubtype="0" fill="hold" '
                  f'nodeType="{"afterEffect" if k == 0 else "withEffect"}"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
                  f'<p:childTnLst><p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr><p:cTn id="{b}" dur="8000" fill="hold"/>'
                  f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:cmd></p:childTnLst></p:cTn></p:par>')
    media = ''
    for spid in sp:
        media += (f'<p:video><p:cMediaNode vol="0" mute="1"><p:cTn id="{nid()}" repeatCount="indefinite" fill="hold" display="0">'
                  f'<p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>'
                  f'</p:cMediaNode></p:video>')
    xml = (f'<p:timing xmlns:p="{NS["p"]}"><p:tnLst><p:par><p:cTn id="{root}" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
           f'<p:seq concurrent="1" nextAc="seek"><p:cTn id="{main}" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
           f'<p:par><p:cTn id="{par3}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="{main}"/></p:cond></p:stCondLst><p:childTnLst>'
           f'<p:par><p:cTn id="{par4}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>{calls}</p:childTnLst></p:cTn></p:par>'
           f'</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn>'
           f'<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
           f'<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>'
           f'{media}</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>')
    el = etree.fromstring(xml)
    ext = s._element.find('{%s}extLst' % NS['p'])
    if ext is not None:
        ext.addprevious(el)
    else:
        s._element.append(el)


# ================================================================ 1. cover
s = S[0]
find(s, 'TextBox 1').text_frame.paragraphs[0].runs[0].text = 'Midterm Presentation'
for r in find(s, 'TextBox 1').text_frame.paragraphs[0].runs[1:]:
    r.text = ''
sub = find(s, 'TextBox 16', 0).text_frame
sub.paragraphs[0].runs[0].text = 'MLAVL 재현 + 1차 개선: 춤 동작 사전(Bailando)을 더한 심판 점수 예측'
for r in sub.paragraphs[0].runs[1:]:
    r.text = ''
for p in sub.paragraphs[1:]:
    p._p.getparent().remove(p._p)
dp = find(s, 'TextBox 2').text_frame.paragraphs[0]
dp.runs[0].text = '2026.  11.  10.'
for r in dp.runs[1:]:
    r.text = ''
find(s, 'TextBox 20').text_frame.paragraphs[0].runs[0].text = 'CVPR 2025'
for k, txt in ((1, 'Base: MLAVL, Xu et al., “Language-Guided Audio-Visual Learning for Long-Term Sports Assessment”'),
               (2, '+ Bailando (Siyao et al., CVPR 2022)의 choreographic memory · Code: github.com/XuHuangbiao/MLAVL')):
    tf = find(s, 'TextBox 16', k).text_frame
    tf.paragraphs[0].runs[0].text = txt
    for r in tf.paragraphs[0].runs[1:]:
        r.text = ''
    for p in tf.paragraphs[1:]:
        p._p.getparent().remove(p._p)
pic = find(s, 'Picture 24'); pic._element.getparent().remove(pic._element)
t = s._element.find('p:timing', NS)
if t is not None:
    s._element.remove(t)
drop_unused_rels(s)
videos(s, [('fisv154_raw', 7.0, 5.45, 2.3, 1.29), ('rg_ribbon010_raw', 9.45, 5.45, 2.3, 1.29)])

# ================================================================ 2. task & approach
s = S[1]; clear(s); title(s, 'Task & Approach')
lead(s, '긴 경기 영상을 보고 심판 점수를 예측하는 MLAVL에, Bailando의 ‘춤 동작 사전’을 더해 개선한다')
ins = [('영상 클립', '1.3초 단위로 자른 경기 영상'), ('음악', '경기 배경음악'),
       ('동작 설명 텍스트', '‘던지고 받기’, ‘나선’ …')]
for k, (h_, l_) in enumerate(ins):
    box(s, 0.44, 1.7 + k * 0.85, 2.7, 0.72, [h_, (l_, {'size': 12.5, 'b': False, 'c': BLUE})], size=14)
    arr(s, 3.22, 1.9 + k * 0.85)
box(s, 0.44, 4.25, 2.7, 0.75, ['+ 선수 포즈의 동작 코드', ('Bailando 동작 사전 (ours)', {'size': 12.5, 'b': False, 'c': '2E7D32'})],
    fill=GREEN, line='2E7D32', dash=True, size=14, color='1B5E20')
arr(s, 3.22, 4.47, color='66BB6A')
box(s, 3.7, 1.7, 2.45, 3.3, ['MLAVL', ('CVPR 2025', {'size': 12, 'b': False, 'c': 'C9D6E3'}), ('', {'size': 8}),
                             ('동작 텍스트 그래프로', {'size': 13.5, 'b': False, 'c': 'FFFFFF'}),
                             ('클립을 안내하고', {'size': 13.5, 'b': False, 'c': 'FFFFFF'}),
                             ('영상·음악을 교차 결합', {'size': 13.5, 'b': False, 'c': 'FFFFFF'})],
    fill=NAVY, size=20, color='FFFFFF')
arr(s, 6.23, 3.18)
box(s, 6.72, 2.75, 1.45, 1.2, ['심판 점수', ('예측', {'size': 15, 'b': True})], fill='FFFFFF', line=NAVY, size=13.5)
panel(s, 8.45, 1.65, 4.7, 3.4)
textbox(s, 8.5, 3.22, 2.05, 0.3, [('리듬체조', {'b': True}), (' · 1,000편', {})], size=12, color=NAVY, align='ctr')
textbox(s, 10.5, 3.22, 2.6, 0.3, [('피겨', {'b': True}), (' · 500편 · 예: 김연아', {})], size=12, color=NAVY, align='ctr')
bullets(s, 8.5, 3.62, 4.6, 1.4, [
    ('s', [('ρ', {'b': True}), (': 예측과 심판 점수의 순위 일치도 (1이 완벽)', {})]),
    ('s', [('MSE', {'b': True}), (': 점수 오차 (낮을수록 좋음)', {})]),
    ('s', [('점수', {'b': True}), (': RG 총점 · 피겨 TES 기술 / PCS 예술', {})]),
], ssz=13, space=250)
videos(s, [('rg_4app_raw', 8.55, 1.75, 1.92, 1.44), ('fisv154_raw', 10.55, 1.75, 2.5, 1.41)])
bar(s, 5.72, '진행 순서', '① 논문 재현  →  ② 개선 모듈을 정답이 확실한 논문 데이터셋에서 먼저 검증  →  ③ 브레이킹 배틀 심사에 적용', size=15)
foot(s, 'MLAVL (CVPR 2025) · Bailando (CVPR 2022) · RG: Zeng et al., MM 2020 · Fis-V: Xu et al., TCSVT 2019 (#154 = 2014 소치 쇼트)')

# ================================================================ 3. reproduction
s = S[2]; clear(s); title(s, 'Reproduction')
lead(s, '논문 설정 그대로 재현: 피겨 예술 점수는 거의 같고, 리듬체조는 논문보다 0.07 낮다')
chart(s, 0.44, 1.55, 7.45, 4.05, ['RG Ball', 'RG Clubs', 'RG Hoop', 'RG Ribbon', 'Fis-V TES', 'Fis-V PCS'],
      [('논문 보고값', (0.826, 0.829, 0.871, 0.866, 0.766, 0.863), GRAYBAR),
       ('재현 (3번 실행 평균)', (0.758, 0.749, 0.786, 0.808, 0.723, 0.852), BLUE)],
      0.60, 0.90, 'Spearman ρ (높을수록 좋음)', fmt='0.00')
textbox(s, 8.15, 1.6, 5.0, 0.4, '격차에 대해 확인한 것', size=16, color=NAVY, bold=True)
causes = [('① 입력·설정은 논문과 같음', '저자와 같은 공개 특징(PAMFN), 논문 하이퍼파라미터', '정규화 실험 96회로도 격차는 그대로'),
          ('② 공개 코드 ≠ 논문 모델', '학습 파라미터 6.99M (논문 3.82M)', '→ 논문 실험과 같은 코드인지 알 수 없음'),
          ('③ 논문은 보고 방식 미기재', '실행 횟수·선택 방식을 밝히지 않음', '우리 최고 1회: Ribbon 0.842 (논문 0.866)')]
for k, (h_, a_, b_) in enumerate(causes):
    box(s, 8.15, 2.05 + k * 1.18, 5.0, 1.06, [h_, (a_, {'size': 13, 'b': False, 'c': BLUE}),
                                               (b_, {'size': 13, 'b': False, 'c': BLUE})], align='l', size=14)
bar(s, 5.72, '기준선', '격차 원인은 미확인 → 이후 비교는 이 재현값(3번 평균 RG 0.775 · Fis-V 0.788)과 같은 조건에서 한다', size=15)
foot(s, 'Paper: Xu et al., CVPR 2025 · 논문과 같은 Fisher z 평균: RG 0.849 → 0.776, Fis-V 0.823 → 0.797 · 그래프는 3번 실행 평균')

# ================================================================ 4. attempt 1
s = S[3]; clear(s); title(s, 'Attempt 1: Feature Codebook')
lead(s, '1차 시도: 영상 특징으로 만든 동작 사전은 효과가 없었다')
panel(s, 0.44, 1.6, 5.75, 1.55)
bullets(s, 0.5, 1.62, 5.65, 1.5, [
    ('h', '왜 논문 데이터셋부터?'),
    ('s', '브레이킹(WDSF 실제 심판 점수)에 먼저 넣었지만, 6가지 모델 모두 단순 규칙과 비슷한 수준'),
    ('s', [('→ 모듈 문제인지 데이터 문제인지 구분 불가', {'c': RED})]),
], hsz=16, ssz=14, space=300)
textbox(s, 0.44, 3.4, 5.75, 0.35, '1차 시도의 구조', size=15, color=NAVY, bold=True)
flow = [('영상', ''), ('Video Swin 특징', 'MLAVL 입력과 같음'), ('동작 사전', '특징 → 번호 128개'), ('MLAVL', '')]
xs = [0.44, 1.5, 3.36, 5.12]; ws = [0.8, 1.6, 1.5, 1.07]
for k, ((h_, l_), x_, w_) in enumerate(zip(flow, xs, ws)):
    lines = [h_] + ([(l_, {'size': 11, 'b': False, 'c': BLUE})] if l_ else [])
    box(s, x_, 3.85, w_, 0.95, lines, size=12.5, fill='FFFFFF' if k in (0, 3) else LIGHT, line='C9D6E3')
    if k < 3:
        arr(s, x_ + w_ + 0.03, 4.18, w=0.2, h=0.28)
textbox(s, 0.44, 4.95, 5.75, 0.35, [('재료가 MLAVL이 이미 보는 특징 → 새 정보가 없다', {'b': True, 'c': RED})], size=13.5)
chart(s, 6.45, 1.55, 6.7, 4.0, ['리듬체조 (RG)', '피겨 (Fis-V)'],
      [('재현 (기준선)', (0.775, 0.788), GRAYBAR),
       ('+ 사전을 그래프 노드로', (0.773, 0.774), MID),
       ('+ 사전을 특징에 더하기', (0.765, 0.767), BLUE)],
      0.70, 0.82, 'Spearman ρ (3번 실행 평균)', unit=0.02, fmt='0.000', lsz=12)
bar(s, 5.72, '원인', '사전을 영상 특징으로 만들면 이미 보는 정보의 반복 → Bailando처럼 ‘몸의 자세(포즈)’로 사전을 만들어야 한다', size=15)
foot(s, 'WDSF: World DanceSport Federation · 단순 규칙: “두 번째로 춘 댄서가 이긴다” · 동작 사전 = VQ-VAE 코드북 (Bailando)')

# ================================================================ 5. attempt 2
s = S[4]; clear(s); title(s, 'Attempt 2: Pose Codebook')
lead(s, '2차 시도: 원본 영상에서 선수의 관절 움직임을 뽑아 Bailando식 동작 사전을 만들었다')
cards = [('① 원본 영상 받기', '리듬체조 1,000편', '피겨 500편 (45 GB)'),
         ('② 선수 자세 추출', '관절 17개 · 초당 8장', '1명 추적 · 프레임 95 – 98%'),
         ('③ 동작 사전 학습', '0.48초 동작 → 번호 1개', '대표 동작 512개 (상체·하체)'),
         ('④ MLAVL에 입력', '1.3초 영상 구간마다', '그 구간 동작 코드의 벡터 추가')]
for k, (h_, a_, b_) in enumerate(cards):
    x_ = 0.44 + k * 3.2
    box(s, x_, 1.6, 2.88, 1.4, [h_, (a_, {'size': 13.5, 'b': False, 'c': BLUE}), (b_, {'size': 13.5, 'b': False, 'c': BLUE})],
        fill=GREEN if k == 2 else LIGHT, size=14)
    if k < 3:
        arr(s, x_ + 2.9, 2.14, w=0.28, h=0.3)
videos(s, [('fisv154_pose', 0.44, 3.15, 4.1, 2.31), ('rg_same191', 4.7, 3.15, 3.08, 2.31)])
panel(s, 7.95, 3.15, 5.2, 2.31)
bullets(s, 8.0, 3.15, 5.1, 2.3, [
    ('h', '영상 읽는 법'),
    ('s', [('왼쪽', {'b': True}), (': 김연아(소치)의 자세 + 지금 동작 번호', {})]),
    ('s', [('가운데', {'b': True}), (': 다른 영상 4편의 비슷한 균형 자세 = 상체 #191', {})]),
    ('h', '사전 품질'),
    ('s', '512개 모두 사용 (실효 사용 수 418 – 463개)'),
    ('s', '복원 오차 = 평균 자세로 대체한 경우의 40 – 60%'),
], hsz=15, ssz=13, space=150)
bar(s, 5.72, '1차와 차이', '영상 특징을 다시 쓰지 않고, 영상 특징에 없는 ‘관절 움직임’을 새 정보로 더한다', size=15)
foot(s, '파랑 상체 사전(머리·팔) · 빨강 하체 사전(골반·다리) · 회색 몸통 · Keypoint R-CNN · Bailando식 2D VQ-VAE · #191은 고른 예시')

# ================================================================ 6. results
s = S[5]; clear(s); title(s, 'Results')
lead(s, '리듬체조는 4종목 모두 올랐고(0.775 → 0.798), 피겨는 변화가 없었다')
chart(s, 0.44, 1.55, 7.3, 4.05, ['Ball', 'Clubs', 'Hoop', 'Ribbon', '평균'],
      [('재현 (기준선)', (0.758, 0.749, 0.786, 0.808, 0.775), GRAYBAR),
       ('+ 포즈 동작 사전', (0.794, 0.766, 0.802, 0.832, 0.798), BLUE)],
      0.70, 0.85, '리듬체조 Spearman ρ (3번 실행 평균)', fmt='0.000')
calls = [('+0.023', '리듬체조 ρ', '0.775 → 0.798'),
         ('5.10 → 4.84', '리듬체조 최저 MSE', '그 시점의 ρ도 0.766 → 0.781'),
         ('±0.00', '피겨 ρ', '0.788 → 0.787, 변화 없음')]
for k, (big, lab, sub_) in enumerate(calls):
    y_ = 1.6 + k * 1.12
    panel(s, 7.95, y_, 5.2, 1.0, GREEN if k < 2 else LIGHT)
    textbox(s, 7.98, y_ + 0.2, 2.4, 0.6, big, size=26 if len(big) < 8 else 19, color=BLUE if k < 2 else GRAY, bold=True, align='ctr')
    textbox(s, 10.4, y_ + 0.14, 2.7, 0.35, lab, size=14.5, color=NAVY, bold=True)
    textbox(s, 10.4, y_ + 0.5, 2.7, 0.35, sub_, size=12.5, color=BLUE)
textbox(s, 7.95, 4.98, 5.2, 0.6, [('넣는 방식이 중요: ', {'b': True}),
                                  ('코드를 그래프 노드로 붙이면 0.778로 효과 없음, 코드 벡터를 영상 토큰에 더할 때만 효과', {})],
        size=13, color=NAVY)
bar(s, 5.72, '주의', '종목별 상승폭(+0.016 – 0.036)은 시드 편차와 비슷한 크기 → 다음 장의 통제 실험으로 원인 확인', size=15)
foot(s, '모든 값: 시드 3개 평균, 학습 중 최고 test ρ / 최저 test MSE (재현과 같은 방식)')

# ================================================================ 7. ablation & plan
s = S[6]; clear(s); title(s, 'Ablation & Plan')
done = len(ABL) >= 7
lead(s, '개선이 정말 포즈 정보 덕분인지 통제 실험으로 확인' + ('' if done else ' 중 (11/6 완료 예정)'))


def v(k, base=0.798):
    if k not in ABL:
        return '진행 중'
    return f'{ABL[k]:.3f} ({ABL[k] - base:+.3f})'


rows = [['확인할 질문', '실험', 'RG ρ (차이)'],
        ['기준: 포즈 동작 코드 사용', '—', '0.798'],
        ['그 영상의 실제 동작이 필요한가?', '다른 영상의 동작 코드로 바꿔치기', v('swap')],
        ['동작 순서가 중요한가?', '같은 영상 안에서 구간 순서 섞기', v('shuffle')],
        ['코드(사전)로 바꾸는 게 필요한가?', '코드 대신 연속 자세 통계 입력', v('posecont')],
        ['상체와 하체 중 무엇이 중요한가?', '상체 코드만 사용', v('upper')],
        ['', '하체 코드만 사용', v('lower')],
        ['기준: 재현 (포즈 없음)', '—', '0.775'],
        ['MLAVL의 음악·텍스트는 필요한가?', '음악 입력 제거', v('no_audio', 0.775)],
        ['', '텍스트 동작 그래프 제거', v('no_text_graph', 0.775)]]
table(s, 0.44, 1.6, 8.15, rows, [3.3, 3.2, 1.65], row_h=0.39, size=13, hl=(1, 7), bold_rows=(1, 7), aligns=['l', 'l', 'c'])
panel(s, 8.85, 1.6, 4.3, 3.96, LIGHT)
bullets(s, 8.9, 1.62, 4.2, 3.9, [
    ('h', '읽는 법'),
    ('s', '기준(0.798)보다 크게 떨어짐 → 그 요소가 개선에 기여'),
    ('s', '재현값(0.775)까지 떨어짐 → 개선 전부가 그 요소 덕분'),
    ('s', '아래 두 줄은 포즈 없이 MLAVL 구성요소만 제거 (기준 0.775)'),
    ('h', '기말 발표까지'),
    ('s', '브레이킹 배틀 심사에 같은 방식 적용'),
    ('s', '피겨 기술 점수(TES)가 안 오르는 원인 분석'),
], hsz=16, ssz=13.5, space=250)
bar(s, 5.72, 'Roadmap', '① 재현 ✓  →  ② 포즈 동작 사전으로 리듬체조 개선 ✓  →  ③ 통제 실험' + (' ✓' if done else '') + '  →  ④ 브레이킹 적용 (기말)', size=15)
foot(s, '모든 실험: 리듬체조 4종목 × 시드 3개, 같은 학습 설정 · 감사합니다')

prs.save(OUT)
print('saved', OUT)
