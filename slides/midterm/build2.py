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
    textbox(s, 0.44, 0.98, 12.7, 0.5, text, size=18, color=NAVY, bold=True)


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
# Slide text is in English (about half of the audience is international); the talk and the script stay in Korean.
s = S[0]
find(s, 'TextBox 1').text_frame.paragraphs[0].runs[0].text = 'Midterm Presentation'
for r in find(s, 'TextBox 1').text_frame.paragraphs[0].runs[1:]:
    r.text = ''
sub = find(s, 'TextBox 16', 0).text_frame
sub.paragraphs[0].runs[0].text = 'Reproducing MLAVL + 1st improvement: a Bailando pose codebook for judge-score prediction'
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
               (2, '+ choreographic memory of Bailando (Siyao et al., CVPR 2022) · Code: github.com/XuHuangbiao/MLAVL')):
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
lead(s, "MLAVL scores long performance videos; we add Bailando's dance-move codebook")
ins = [('Video clips', '1.3 s routine segments'), ('Music', 'background music'),
       ('Action text', '“throw and catch”, “spirals”')]
for k, (h_, l_) in enumerate(ins):
    box(s, 0.44, 1.7 + k * 0.85, 2.7, 0.72, [h_, (l_, {'size': 12.5, 'b': False, 'c': BLUE})], size=14)
    arr(s, 3.22, 1.9 + k * 0.85)
box(s, 0.44, 4.25, 2.7, 0.75, ['+ Pose codes', ('Bailando codebook (ours)', {'size': 12.5, 'b': False, 'c': '2E7D32'})],
    fill=GREEN, line='2E7D32', dash=True, size=14, color='1B5E20')
arr(s, 3.22, 4.47, color='66BB6A')
box(s, 3.7, 1.7, 2.45, 3.3, ['MLAVL', ('CVPR 2025', {'size': 12, 'b': False, 'c': 'C9D6E3'}), ('', {'size': 8}),
                             ('action-text graph', {'size': 13.5, 'b': False, 'c': 'FFFFFF'}),
                             ('guides the clips;', {'size': 13.5, 'b': False, 'c': 'FFFFFF'}),
                             ('video × music fusion', {'size': 13.5, 'b': False, 'c': 'FFFFFF'})],
    fill=NAVY, size=20, color='FFFFFF')
arr(s, 6.23, 3.18)
box(s, 6.72, 2.75, 1.45, 1.2, ['Judge score', ('prediction', {'size': 15, 'b': True})], fill='FFFFFF', line=NAVY, size=13.5)
panel(s, 8.45, 1.65, 4.7, 3.4)
textbox(s, 8.5, 3.22, 2.05, 0.3, [('Gymnastics', {'b': True}), (' · 1,000', {})], size=11.5, color=NAVY, align='ctr')
textbox(s, 10.5, 3.22, 2.6, 0.3, [('Skating', {'b': True}), (' · 500 (Kim Yuna)', {})], size=11.5, color=NAVY, align='ctr')
bullets(s, 8.5, 3.62, 4.6, 1.4, [
    ('s', [('ρ', {'b': True}), (': rank match with judges (1 = best)', {})]),
    ('s', [('MSE', {'b': True}), (': score error (lower = better)', {})]),
    ('s', [('Scores', {'b': True}), (': RG total; skating TES / PCS', {})]),
], ssz=13, space=250)
videos(s, [('rg_4app_raw', 8.55, 1.75, 1.92, 1.44), ('fisv154_raw', 10.55, 1.75, 2.5, 1.41)])
bar(s, 5.72, 'Plan', '① Reproduce  →  ② Validate the module on the paper datasets first  →  ③ Apply to breaking battles', size=15)
foot(s, 'MLAVL (CVPR 25) · Bailando (CVPR 22) · RG: Zeng et al., MM 20 · Fis-V: Xu et al., TCSVT 19 (#154 = Sochi 2014 short program)')

# ================================================================ 3. reproduction
s = S[2]; clear(s); title(s, 'Reproduction')
lead(s, 'Reproduced with paper settings: skating PCS matches, gymnastics is 0.07 lower')
chart(s, 0.44, 1.55, 7.45, 4.05, ['RG Ball', 'RG Clubs', 'RG Hoop', 'RG Ribbon', 'Fis-V TES', 'Fis-V PCS'],
      [('Paper', (0.826, 0.829, 0.871, 0.866, 0.766, 0.863), GRAYBAR),
       ('Ours (mean of 3 runs)', (0.758, 0.749, 0.786, 0.808, 0.723, 0.852), BLUE)],
      0.60, 0.90, 'Spearman ρ (higher is better)', fmt='0.00')
textbox(s, 8.15, 1.6, 5.0, 0.4, 'What we know about the gap', size=16, color=NAVY, bold=True)
causes = [('① Same inputs and settings', 'Same public features (PAMFN), paper hyper-params', '96 regularisation runs did not close it'),
          ('② Released code ≠ paper model', 'Trainable params 6.99M (paper: 3.82M)', "→ unclear if it is the paper's code"),
          ('③ Reporting not specified', 'The paper gives no #runs or selection rule', 'Our best single run: Ribbon 0.842 (paper 0.866)')]
for k, (h_, a_, b_) in enumerate(causes):
    box(s, 8.15, 2.05 + k * 1.18, 5.0, 1.06, [h_, (a_, {'size': 13, 'b': False, 'c': BLUE}),
                                               (b_, {'size': 13, 'b': False, 'c': BLUE})], align='l', size=14)
bar(s, 5.72, 'Baseline', 'Gap cause unknown → we compare against our reproduction: RG 0.775 · Fis-V 0.788', size=15)
foot(s, 'Paper: Xu et al., CVPR 2025 · Fisher-z means as in the paper: RG 0.849 → 0.776, Fis-V 0.823 → 0.797 · bars: mean of 3 runs')

# ================================================================ 4. attempt 1
s = S[3]; clear(s); title(s, 'Attempt 1: Feature Codebook')
lead(s, 'Attempt 1: a codebook built on video features did not help')
panel(s, 0.44, 1.6, 5.75, 1.55)
bullets(s, 0.5, 1.62, 5.65, 1.5, [
    ('h', 'Why the paper datasets first?'),
    ('s', 'On breaking (real WDSF judge scores), all 6 models were close to a trivial rule'),
    ('s', [('→ module problem or data problem? Could not tell', {'c': RED})]),
], hsz=16, ssz=14, space=300)
textbox(s, 0.44, 3.4, 5.75, 0.35, 'Structure of attempt 1', size=15, color=NAVY, bold=True)
flow = [('Video', ''), ('Video Swin feature', '= MLAVL input'), ('Codebook', 'feature → 128 codes'), ('MLAVL', '')]
xs = [0.44, 1.42, 3.3, 5.08]; ws = [0.86, 1.66, 1.56, 1.11]
for k, ((h_, l_), x_, w_) in enumerate(zip(flow, xs, ws)):
    lines = [h_] + ([(l_, {'size': 11, 'b': False, 'c': BLUE})] if l_ else [])
    box(s, x_, 3.85, w_, 0.95, lines, size=12.5, fill='FFFFFF' if k in (0, 3) else LIGHT, line='C9D6E3')
    if k < 3:
        arr(s, x_ + w_ + 0.03, 4.18, w=0.2, h=0.28)
textbox(s, 0.44, 4.95, 5.75, 0.35, [('Made from what MLAVL already sees → nothing new', {'b': True, 'c': RED})], size=13.5)
chart(s, 6.45, 1.55, 6.7, 4.0, ['Rhythmic gym. (RG)', 'Figure skating (Fis-V)'],
      [('Reproduction (baseline)', (0.775, 0.788), GRAYBAR),
       ('+ codes as graph nodes', (0.773, 0.774), MID),
       ('+ codes added to features', (0.765, 0.767), BLUE)],
      0.70, 0.82, 'Spearman ρ (mean of 3 runs)', unit=0.02, fmt='0.000', lsz=12)
bar(s, 5.72, 'Why', 'Re-coding the same features adds nothing → build the codebook from body pose, as Bailando does', size=15)
foot(s, 'WDSF: World DanceSport Federation · trivial rule: “the second dancer wins” · codebook = VQ-VAE (Bailando)')

# ================================================================ 5. attempt 2
s = S[4]; clear(s); title(s, 'Attempt 2: Pose Codebook')
lead(s, "Attempt 2: a Bailando-style codebook of the athlete's joint motion")
cards = [('① Raw videos', 'Rhythmic gym. 1,000', 'Figure skating 500 (45 GB)'),
         ('② Athlete pose', '17 joints · 8 per second', '1 person tracked · 95 – 98%'),
         ('③ Pose codebook', '0.48 s of motion → 1 code', '512 upper + 512 lower'),
         ('④ Into MLAVL', 'for every 1.3 s clip', 'add its code vectors')]
for k, (h_, a_, b_) in enumerate(cards):
    x_ = 0.44 + k * 3.2
    box(s, x_, 1.6, 2.88, 1.4, [h_, (a_, {'size': 13.5, 'b': False, 'c': BLUE}), (b_, {'size': 13.5, 'b': False, 'c': BLUE})],
        fill=GREEN if k == 2 else LIGHT, size=14)
    if k < 3:
        arr(s, x_ + 2.9, 2.14, w=0.28, h=0.3)
videos(s, [('fisv154_pose', 0.44, 3.15, 4.1, 2.31), ('rg_same191', 4.7, 3.15, 3.08, 2.31)])
panel(s, 7.95, 3.15, 5.2, 2.31)
bullets(s, 8.0, 3.15, 5.1, 2.3, [
    ('h', 'How to read the videos'),
    ('s', [('Left', {'b': True}), (': Kim Yuna (Sochi), pose + current codes', {})]),
    ('s', [('Middle', {'b': True}), (': similar poses, 4 videos = #191', {})]),
    ('h', 'Codebook quality'),
    ('s', 'All 512 codes used (perplexity 418 – 463)'),
    ('s', 'Reconstruction error 40 – 60% of mean pose'),
], hsz=15, ssz=13, space=150)
bar(s, 5.72, 'vs. attempt 1', "Instead of re-coding video features, add new information: the athlete's joint motion", size=15)
foot(s, 'Blue: upper codebook (head, arms) · red: lower (hips, legs) · grey: torso · Bailando-style 2D VQ-VAE · #191 hand-picked')

# ================================================================ 6. results
s = S[5]; clear(s); title(s, 'Results')
lead(s, 'Gymnastics improved on all 4 apparatus (0.775 → 0.798); skating did not change')
chart(s, 0.44, 1.55, 7.3, 4.05, ['Ball', 'Clubs', 'Hoop', 'Ribbon', 'Mean'],
      [('Reproduction (baseline)', (0.758, 0.749, 0.786, 0.808, 0.775), GRAYBAR),
       ('+ pose codebook', (0.794, 0.766, 0.802, 0.832, 0.798), BLUE)],
      0.70, 0.85, 'Rhythmic gymnastics, Spearman ρ (mean of 3 runs)', fmt='0.000')
calls = [('+0.023', 'RG ρ', '0.775 → 0.798'),
         ('5.10 → 4.84', 'RG best MSE', 'ρ there: 0.766 → 0.781'),
         ('±0.00', 'Skating ρ', '0.788 → 0.787, no change')]
for k, (big, lab, sub_) in enumerate(calls):
    y_ = 1.6 + k * 1.12
    panel(s, 7.95, y_, 5.2, 1.0, GREEN if k < 2 else LIGHT)
    textbox(s, 7.98, y_ + 0.2, 2.4, 0.6, big, size=26 if len(big) < 8 else 19, color=BLUE if k < 2 else GRAY, bold=True, align='ctr')
    textbox(s, 10.4, y_ + 0.14, 2.7, 0.35, lab, size=14.5, color=NAVY, bold=True)
    textbox(s, 10.4, y_ + 0.5, 2.7, 0.35, sub_, size=12.5, color=BLUE)
textbox(s, 7.95, 4.98, 5.2, 0.6, [('How codes enter matters: ', {'b': True}),
                                  ('graph nodes 0.778 (no gain); adding to video tokens helps', {})],
        size=13, color=NAVY)
bar(s, 5.72, 'Check', 'Codes from another video erase the gain (0.772, p = 0.005) → it comes from the athlete\'s own motion', size=15)
foot(s, 'All values: mean of 3 seeds; best test ρ and best test MSE during training (same protocol as the reproduction)')

# ================================================================ 7. ablation & plan
s = S[6]; clear(s); title(s, 'Ablation & Plan')
KEYS = ('swap', 'shuffle', 'posecont', 'upper', 'lower', 'no_audio', 'no_text_graph')
done = all(k in ABL for k in KEYS)
PV = ABL.get('p', {})
lead(s, ("The gain needs the athlete's own, time-aligned lower-body motion" if done
         else 'Is the gain really from pose? Control experiments (running, done 11/6)'))


def v(k, base=0.798):
    if k not in ABL:
        return 'running'
    return f'{ABL[k]:.3f} ({ABL[k] - base:+.3f})'


def pv(k):
    return '' if k not in PV else (f'{PV[k]:.3f}' if PV[k] < 0.1 else f'{PV[k]:.2f}')


rows = [['Question', 'Experiment', 'RG ρ (Δ)', 'p'],
        ['Baseline: pose codes', '—', '0.798', ''],
        ["Is the video's own motion needed?", 'Codes from another video', v('swap'), pv('swap')],
        ['Does the timing matter?', 'Shuffle clips within the video', v('shuffle'), pv('shuffle')],
        ['Is the codebook needed?', 'Continuous pose statistics instead', v('posecont'), pv('posecont')],
        ['Upper or lower body?', 'Upper-body codes only', v('upper'), pv('upper')],
        ['', 'Lower-body codes only', v('lower'), pv('lower')],
        ['Baseline: reproduction', '—', '0.775', ''],
        ["Are MLAVL's music / text needed?", 'Remove the music input', v('no_audio', 0.775), pv('no_audio')],
        ['', 'Remove the action-text graph', v('no_text_graph', 0.775), pv('no_text_graph')]]
table(s, 0.44, 1.6, 8.15, rows, [2.95, 2.85, 1.6, 0.75], row_h=0.39, size=13, hl=(1, 7), bold_rows=(1, 7), aligns=['l', 'l', 'c', 'c'])
panel(s, 8.85, 1.6, 4.3, 3.96, LIGHT)
bullets(s, 8.9, 1.62, 4.2, 3.9, [
    ('h', 'What we learned'),
    ('s', "Own video, right timing: other video / shuffled → gain lost"),
    ('s', 'Lower body (hips, legs) carries the gain'),
    ('s', 'Continuous pose works as well as codes → pose is the key; codes give readable motion units'),
    ('s', "MLAVL's music and text graph: no measurable effect here"),
    ('h', 'Until the final'),
    ('s', 'Bailando Beat Align as input (running) · breaking battles'),
], hsz=15, ssz=12.5, space=180)
bar(s, 5.72, 'Roadmap', '① Reproduction ✓  →  ② Pose codebook improves RG ✓  →  ③ Controls' + (' ✓' if done else '') + '  →  ④ Breaking (final)', size=15)
foot(s, 'All: 4 RG apparatus × 3 seeds, same settings · p: paired t-test over the 12 runs vs. the bold baseline · Thank you')

prs.save(OUT)
print('saved', OUT)
