"""Template helpers for the midterm deck: loads the 2nd paper-review deck and copies its styled shapes (panels, bars,
bullets, tables) so new slides keep the 1st-deck look."""
import copy
import json
import sys

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.util import Inches, Pt, Emu

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # all paths below are relative to slides/midterm
SRC = '../../presentations/2025198001_정현도_논문발표2.pptx'   # template = 2nd paper-review deck (local only)
NS = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
A = '{%s}' % NS['a']
NAVY, BLUE, RED, GRAY = '0B2C4A', '1F497D', 'C0504D', '7F7F7F'
LIGHT, GREEN = 'E8F0F5', 'D4EEDA'

prs = Presentation(SRC)
S = list(prs.slides)


def find(slide, name, nth=0):
    return [sh for sh in slide.shapes if sh.name == name][nth]


# exemplar elements (copied before anything is removed)
E_PANEL = copy.deepcopy(find(S[3], 'Rounded Rectangle 16')._element)
E_BAR = copy.deepcopy(find(S[1], 'Rounded Rectangle 19')._element)
E_BARTXT = copy.deepcopy(find(S[1], 'TextBox 20')._element)
E_FOOT = copy.deepcopy(find(S[1], 'TextBox 21')._element)
E_BUL = copy.deepcopy(find(S[1], '직사각형 16')._element)
P_HEAD, P_SUB = [copy.deepcopy(p) for p in E_BUL.findall('.//a:p', NS)[:2]]
CHROME = {'직사각형 9', 'TextBox 12', '그림 13', 'TextBox 14', '그림 3'}


def next_id(slide):
    ids = [int(e.get('id')) for e in slide._element.iter('{%s}cNvPr' % NS['p'])]
    return max(ids + [1]) + 1


def place(slide, el, x, y, w, h):
    el = copy.deepcopy(el)
    el.find('.//p:cNvPr', NS).set('id', str(next_id(slide)))
    off, ext = el.find('.//a:xfrm/a:off', NS), el.find('.//a:xfrm/a:ext', NS)
    off.set('x', str(int(Inches(x)))); off.set('y', str(int(Inches(y))))
    ext.set('cx', str(int(Inches(w)))); ext.set('cy', str(int(Inches(h))))
    slide.shapes._spTree.append(el)
    return el


def set_fill(el, color):
    el.find('.//a:solidFill/a:srgbClr', NS).set('val', color)


def panel(slide, x, y, w, h, color=LIGHT, adj=5000):
    el = place(slide, E_PANEL, x, y, w, h)
    set_fill(el, color)
    el.find('.//a:gd', NS).set('fmla', f'val {adj}')
    return el


def runs_xml(p, segs, size, color=None, bold=None):
    """segs: str or list of (text, opts) ; opts keys b, c"""
    proto = p.find('a:r', NS)
    for r in p.findall('a:r', NS):
        p.remove(r)
    if isinstance(segs, str):
        segs = [(segs, {})]
    for text, o in segs:
        r = copy.deepcopy(proto)
        rpr = r.find('a:rPr', NS)
        rpr.set('sz', str(int(size * 100)))
        b = o.get('b', bold)
        if b is not None:
            rpr.set('b', '1' if b else '0')
        c = o.get('c', color)
        if c:
            sf = rpr.find('a:solidFill', NS)
            for ch in list(sf):
                sf.remove(ch)
            etree.SubElement(sf, A + 'srgbClr').set('val', c)
        r.find('a:t', NS).text = text
        end = p.find('a:endParaRPr', NS)
        if end is not None:
            end.addprevious(r)
        else:
            p.append(r)


def bullets(slide, x, y, w, h, items, hsz=20, ssz=15, space=None):
    """items: list of ('h', segs) heading or ('s', segs) sub-bullet"""
    el = place(slide, E_BUL, x, y, w, h)
    body = el.find('.//p:txBody', NS)
    for p in body.findall('a:p', NS):
        body.remove(p)
    for kind, segs in items:
        p = copy.deepcopy(P_HEAD if kind == 'h' else P_SUB)
        runs_xml(p, segs, hsz if kind == 'h' else ssz)
        if space is not None:
            p.find('a:pPr/a:spcBef/a:spcPts', NS).set('val', str(space if kind == 's' else space + 300))
        body.append(p)
    return el


def bar(slide, y, label, text, h=0.6, size=16, x=0.44, w=12.45):
    place(slide, E_BAR, x, y, w, h)
    el = place(slide, E_BARTXT, x + 0.15, y, w - 0.3, h)
    p = el.find('.//a:p', NS)
    rs = p.findall('a:r', NS)
    rs[0].find('a:t', NS).text = label + '   '
    rs[1].find('a:t', NS).text = text
    for r in rs:
        r.find('a:rPr', NS).set('sz', str(size * 100))


def foot(slide, text):
    el = place(slide, E_FOOT, 1.0, 6.48, 10.6, 0.3)
    el.find('.//a:t', NS).text = text


def textbox(slide, x, y, w, h, segs, size=14, color=BLUE, align='l', bold=False):
    el = place(slide, E_FOOT, x, y, w, h)
    p = el.find('.//a:p', NS)
    p.find('a:pPr', NS).set('algn', align)
    runs_xml(p, segs, size, color=color, bold=bold)
    return el


def table(slide, x, y, w, rows, widths, row_h=0.3, size=12, hl=(), bold_rows=(), dim_cols=(), aligns=None):
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gf.table
    tot = sum(widths)
    for j, wd in enumerate(widths):
        tbl.columns[j].width = Emu(int(Inches(w) * wd / tot))
    for i in range(nr):
        tbl.rows[i].height = Inches(row_h)
        for j in range(nc):
            cell = tbl.cell(i, j)
            txt = rows[i][j]
            cell.margin_top = cell.margin_bottom = Emu(18288)
            cell.margin_left = cell.margin_right = Emu(54864)
            cell.vertical_anchor = 3  # middle
            tf = cell.text_frame
            tf.text = ''
            p = tf.paragraphs[0]
            p.alignment = 2 if (aligns is None or aligns[j] == 'c') else 1
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(size)
            r.font.name = 'Calibri'
            if i == 0:
                r.font.bold = True
                r.font.color.rgb = RGBColor.from_string('FFFFFF')
                fill = NAVY
            else:
                r.font.bold = i in bold_rows
                r.font.color.rgb = RGBColor.from_string('9AA0A6' if j in dim_cols else '1A1A1A')
                fill = GREEN if i in hl else ('FFFFFF' if i % 2 else 'F4F7FA')
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(fill)
    tblPr = gf._element.find('.//a:tblPr', NS)
    tblPr.set('bandRow', '0')
    return gf


def clear(slide):
    keep_ids = set()
    for sh in list(slide.shapes):
        if sh.name in CHROME:
            continue
        sh._element.getparent().remove(sh._element)
    t = slide._element.find('p:timing', NS)
    if t is not None:
        slide._element.remove(t)
    drop_unused_rels(slide)


def drop_unused_rels(slide):
    xml = etree.tostring(slide._element).decode()
    for rId, rel in list(slide.part.rels.items()):
        if rel.reltype.endswith('/slideLayout') or rel.reltype.endswith('/notesSlide'):
            continue
        if f'"{rId}"' not in xml:
            slide.part.drop_rel(rId)


def title(slide, text):
    tb = find(slide, 'TextBox 12')
    tb.text_frame.paragraphs[0].runs[0].text = text


def arrow(slide, x, y, w=0.35, h=0.35, color=GRAY):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(color)
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def card(slide, x, y, w, h, head, lines, color=LIGHT, hsz=15, lsz=12.5, num=None):
    panel(slide, x, y, w, h, color)
    textbox(slide, x + 0.08, y + 0.08, w - 0.16, 0.4, head, size=hsz, color=NAVY, bold=True, align='ctr')
    for k, ln in enumerate(lines):
        textbox(slide, x + 0.08, y + 0.52 + k * 0.34, w - 0.16, 0.32, ln, size=lsz, color=BLUE, align='ctr')

