"""
Generate IEEE conference Word document for the ACMF fungal classification paper.
Matches the A4 two-column IEEE conference template (conference-template-a4).
"""

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy
from lxml import etree

doc = Document()

# ── Page setup: A4, IEEE margins ─────────────────────────────────────────────
section = doc.sections[0]
section.page_width  = Cm(21.0)
section.page_height = Cm(29.7)
section.top_margin    = Cm(1.9)
section.bottom_margin = Cm(2.54)
section.left_margin   = Cm(1.54)
section.right_margin  = Cm(1.54)
section.header_distance = Cm(1.27)
section.footer_distance = Cm(1.27)

# ── Two-column layout ─────────────────────────────────────────────────────────
def set_two_columns(section):
    sectPr = section._sectPr
    cols = OxmlElement('w:cols')
    cols.set(qn('w:num'), '2')
    cols.set(qn('w:space'), '567')   # ~1 cm gap in twips (567 twips ≈ 1 cm)
    cols.set(qn('w:equalWidth'), '1')
    sectPr.append(cols)

set_two_columns(section)

# ── Helper: set paragraph spacing ────────────────────────────────────────────
def set_para_spacing(para, before=0, after=0, line=None, line_rule=None):
    pPr = para._p.get_or_add_pPr()
    spacing = OxmlElement('w:spacing')
    spacing.set(qn('w:before'), str(before))
    spacing.set(qn('w:after'),  str(after))
    if line:
        spacing.set(qn('w:line'), str(line))
        spacing.set(qn('w:lineRule'), line_rule or 'auto')
    pPr.append(spacing)

def set_font(run, name='Times New Roman', size=10, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)
    # Force Times New Roman for East Asian too
    rPr = run._r.get_or_add_rPr()
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'), name)
    rFonts.set(qn('w:hAnsi'), name)
    rFonts.set(qn('w:cs'), name)
    rPr.insert(0, rFonts)

def add_text_para(text, style_name='Normal', bold=False, italic=False,
                  size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                  before=0, after=60, first_indent=None, left_indent=None):
    para = doc.add_paragraph()
    para.alignment = align
    run = para.add_run(text)
    set_font(run, size=size, bold=bold, italic=italic)
    set_para_spacing(para, before=before, after=after)
    pPr = para._p.get_or_add_pPr()
    ind = OxmlElement('w:ind')
    if first_indent is not None:
        ind.set(qn('w:firstLine'), str(first_indent))
    if left_indent is not None:
        ind.set(qn('w:left'), str(left_indent))
    if first_indent or left_indent:
        pPr.append(ind)
    return para, run

# ── Title (single-column span via section break trick) ───────────────────────
# We'll put the title, authors, abstract, keywords in a single-column header section,
# then switch to two-column for the body.

# Actually: easier to put title/authors BEFORE the 2-col section
# We do this by adding a continuous section break after keywords.

def add_single_col_header():
    """The title, authors, abstract, keywords sit in a 1-col preamble."""
    pass  # We handle this via section properties below

# IEEE copyright line (footnote style at bottom of col1 page1 - approximate with small para)
copy_para = doc.add_paragraph()
copy_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
r = copy_para.add_run('XXX-X-XXXX-XXXX-X/XX/$XX.00 ©2025 IEEE')
set_font(r, size=8)
set_para_spacing(copy_para, before=0, after=40)

# ── TITLE ────────────────────────────────────────────────────────────────────
title_para = doc.add_paragraph()
title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title_para.add_run(
    'Adaptive Confidence-Guided Multi-Scale Vision Transformer\n'
    'Framework for Explainable Microscopic Fungal Species Classification'
)
set_font(r, size=24, bold=False)
set_para_spacing(title_para, before=120, after=80)

# ── AUTHORS (3-column table for author blocks) ────────────────────────────────
auth_table = doc.add_table(rows=1, cols=3)
auth_table.alignment = WD_TABLE_ALIGNMENT.CENTER
auth_table.style = 'Table Grid'
# Remove borders
for row in auth_table.rows:
    for cell in row.cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcBorders = OxmlElement('w:tcBorders')
        for border in ['top','left','bottom','right','insideH','insideV']:
            b = OxmlElement(f'w:{border}')
            b.set(qn('w:val'), 'nil')
            tcBorders.append(b)
        tcPr.append(tcBorders)

def fill_author_cell(cell, lines):
    cell.paragraphs[0].clear()
    for i, line in enumerate(lines):
        if i == 0:
            p = cell.paragraphs[0]
        else:
            p = cell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(line)
        italic = i > 0
        set_font(r, size=9, italic=italic)
        set_para_spacing(p, before=0, after=0)

fill_author_cell(auth_table.rows[0].cells[0], [
    'Rishi Jayanath A',
    'Dept. of Computer Science and Engineering',
    'Karunya Institute of Technology and Sciences',
    'Coimbatore, India',
    'rishi.jayanath@karunya.edu',
])
fill_author_cell(auth_table.rows[0].cells[1], [
    '[Co-author 2 Name]',
    '[Department]',
    '[Institution]',
    '[City, Country]',
    '[email]',
])
fill_author_cell(auth_table.rows[0].cells[2], [
    '[Co-author 3 Name]',
    '[Department]',
    '[Institution]',
    '[City, Country]',
    '[email]',
])

doc.add_paragraph()  # spacer

# ── ABSTRACT ─────────────────────────────────────────────────────────────────
abs_para = doc.add_paragraph()
abs_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r_label = abs_para.add_run('Abstract—')
set_font(r_label, size=9, bold=True, italic=True)
r_body = abs_para.add_run(
    'Pathogenic fungi are not easily identified in the wild. Pathogenic fungi are '
    'difficult to identify in the field. The task of making scopic slides is still '
    'primarily one that requires training from human eyes — a point of delay in '
    'clinical decisions and an introduction of inter-observer error. In this paper '
    'we present an Adaptive We propose a new confidence-guided multi-scale fusion '
    'framework called as ACMF. It uses two encoders, namely runs two Vision '
    'Transformer (ViT-B/16) encoders at different Input resolutions — 224 x 384 '
    'pixels — and merges together their Produces four different ways of fusing '
    'its outputs. These range from simple averaging (S1) and accuracy-weighted '
    'blending (AW1) methods. from simple averaging (S1) and accuracy-weighted '
    'blending (AW1) techniques. The novel per-image entropy-based weighting scheme '
    '(S3) was compared to a typical watermarking method (S2). and an expert MLP '
    'fusion head (S4). Experiments on the The DeFungi benchmark is for five '
    'clinically relevant fungal The accuracy of each species identification was '
    'demonstrated with 6,801 microscopic images. The best weight per-ent variant '
    'is S2 with a macro F1 of 91.63% on the held-out test set — a 3.16 point '
    'gain over ViT-B/16-224 used alone. We also generate maps at inference time '
    '— Attention Rollout. indicate on the model the structures of the hyphae and '
    'spores. handles without any additional markings. Comparisons The rest of the '
    'lineup consists of with ResNet-50, EfficientNet-B3 and Swin-Tiny. evaluation '
    'and explain how the ACMF approach is the most beneficial value.'
)
set_font(r_body, size=9, italic=False)
set_para_spacing(abs_para, before=40, after=40)

# ── KEYWORDS ─────────────────────────────────────────────────────────────────
kw_para = doc.add_paragraph()
kw_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r_kl = kw_para.add_run('Keywords—')
set_font(r_kl, size=9, bold=True, italic=True)
r_kv = kw_para.add_run(
    'fungal classification, vision transformer, multi-task learning, deep learning, '
    'deep vision. Nearly every field needs to understand how to leverage AI-driven '
    'tools such as explainable AI, a fusion of several scales, and attention '
    'rollout. DeFungi, microscopy'
)
set_font(r_kv, size=9, italic=True)
set_para_spacing(kw_para, before=0, after=80)

# ── Switch to two-column layout (insert continuous section break) ─────────────
def insert_section_break_continuous(doc):
    """Insert a continuous section break to start two-column layout."""
    para = doc.add_paragraph()
    set_para_spacing(para, before=0, after=0)
    pPr = para._p.get_or_add_pPr()
    sectPr_new = OxmlElement('w:sectPr')
    cols = OxmlElement('w:cols')
    cols.set(qn('w:num'), '2')
    cols.set(qn('w:space'), '567')
    cols.set(qn('w:equalWidth'), '1')
    sectPr_new.append(cols)
    pgSz = OxmlElement('w:pgSz')
    pgSz.set(qn('w:w'), str(int(21.0 / 2.54 * 1440)))
    pgSz.set(qn('w:h'), str(int(29.7 / 2.54 * 1440)))
    sectPr_new.append(pgSz)
    pgMar = OxmlElement('w:pgMar')
    pgMar.set(qn('w:top'),    str(int(1.9  / 2.54 * 1440)))
    pgMar.set(qn('w:bottom'), str(int(2.54 / 2.54 * 1440)))
    pgMar.set(qn('w:left'),   str(int(1.54 / 2.54 * 1440)))
    pgMar.set(qn('w:right'),  str(int(1.54 / 2.54 * 1440)))
    sectPr_new.append(pgMar)
    pPr.append(sectPr_new)
    return para

insert_section_break_continuous(doc)

# ── Heading helpers ───────────────────────────────────────────────────────────
_section_counter = [0]
_subsection_counter = [0]

def add_section_heading(title):
    _section_counter[0] += 1
    _subsection_counter[0] = 0
    num = ['I','II','III','IV','V','VI','VII','VIII'][_section_counter[0]-1]
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = para.add_run(f'{num}. {title.upper()}')
    set_font(r, size=10, bold=False)
    set_para_spacing(para, before=100, after=40)
    return para

def add_subsection_heading(title):
    _subsection_counter[0] += 1
    letter = 'ABCDEFGH'[_subsection_counter[0]-1]
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = para.add_run(f'{letter}. {title}')
    set_font(r, size=10, italic=True)
    set_para_spacing(para, before=60, after=20)
    return para

def body(text, first_indent=True):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = para.add_run(text)
    set_font(run, size=10)
    set_para_spacing(para, before=0, after=40)
    if first_indent:
        pPr = para._p.get_or_add_pPr()
        ind = OxmlElement('w:ind')
        ind.set(qn('w:firstLine'), '360')  # ~0.6 cm
        pPr.append(ind)
    return para

def equation_para(label, eq_text):
    """Approximate equation as centred text with right-aligned number."""
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = para.add_run(eq_text + f'    ({label})')
    set_font(r, size=10, italic=True)
    set_para_spacing(para, before=40, after=40)
    return para

# ── SECTION I: INTRODUCTION ───────────────────────────────────────────────────
add_section_heading('Introduction')

body(
    'About 1.5 million people per year are killed by fungal infections. and have '
    'an impact on more than a billion [1]. What makes this burden? What makes this '
    'burden? What is particularly frustrating is the majority of the species '
    'responsible are… are well-known. Five of them are the most prevalent '
    'clinical dermatophytes. Complete blood count (CBC): Leucocytes, lymphocytes, '
    'neutrophils, eosinophils, platelets, and blood cells niger, Trichophyton '
    'rubrum, Trichophyton mentagrophytes, and Epidermophyton floccosum. They each '
    'have a different antifungal agent, and it is not only incorrect diagnosis that '
    'is a problem. An academic issue; has direct treatment implications.'
)
body(
    'The traditional approach is direct microscopic examination, which is examined '
    'by a trained microscopist who examines a stained specimen. distinguishes '
    'morphological clues — branching pattern of hyphae, The size of the spores '
    'and the arrangement of spores, the shape of the conidial heads — '
    'Differentiate between species. Works, but will be slow, Requires several years '
    'of practice and any mycologist will Have a disagreement with challenging '
    'objects [2]. Deep learning has made However, there are not any significant '
    'advances in related problems in pathology that have been made by real '
    'application of the method. It is subtlety to microscopic fungi that has been '
    'easy to Overlook: the features to be considered are distributed over a number '
    'of spatial scales at once. They are relatively coarse, with budding clusters '
    'that look like yeast in the shape of buds. and easily visible at normal '
    'resolution; fine hyphal wall It is only possible to resolve the thicknesses '
    'and septum spacings at. higher resolution. A model trained at a single fixed '
    'scale should A compromise between these two regimes.'
)
body(
    'Here, Vision Transformers [4] are a logical choice. Their global '
    'Self-attention may be able to capture long-range spatial regularities that '
    'The word vanishes from the vocabulary of convolutional filters and patch-based '
    'tokenisation produces new words. the resolution a tunable parameter. Although '
    'even ViT, if used at a Single scale, leaves complementary information in the '
    'table. This is the starting point of our work.'
)
body(
    'The Adaptive Confidence-Guided Multi- is a new algorithm for solving the '
    'rotational pose estimation problem. The framework for scale fusion (ACMF) is '
    'used with the ViT-B/16 model at If the 224 and 384 pixels, then merges the '
    'two sets of classes probabilities. Four strategies are studied — simple '
    'averaging Additional validation accuracy weighting (S2), per-image entropy '
    '(S3). A learned MLP gate (S4) — so that the comparison is an informative '
    'about what kind of Combine the actual task the student needs to perform. In '
    'addition to accuracy, S3 generates an explicit image level confidence signal '
    'to be given to a clinician. Which resolution was more informative to the model '
    'for each specimen. Additionally, we have also included Attention Rollout maps '
    '[8] All predictions were based on recognisable criteria. morphological '
    'structures.'
)

# ── SECTION II: RELATED WORK ──────────────────────────────────────────────────
add_section_heading('Related Work')
add_subsection_heading('Bacteria, viruses, fungi and dermatological image classification')
body(
    'Before deep learning took hold, automated fungal iden- Hand-crafted texture '
    'descriptors like were used for tification. LBP and GLCM classifiers were '
    'combined with SVM classifiers. These pipelines needed a lot of feature '
    'engineering and was generalised badly. across staining protocols. '
    'Transfer-learned CNNs changed Significantly reduce the picture: ResNet [5] '
    'and EfficientNet [6] showed that large-scale pretraining and task-specific '
    'fine-tuning could be very effective. specific fine-tuning might be a great '
    'idea on skin lesion Benchmarks and then on microscopic fungal data sets. '
    'The Cassagne et al. DeFungi [10] showed that: The transfer-learned CNNs '
    'could distinguish the five target species with In a fair degree of accuracy, '
    'but there is some confusion between Aspergillus However, the and genera of '
    'Trichophyton continued to be a nuisance — The gap which we directly '
    'tackle here.'
)
add_subsection_heading('AI in Healthcare Marketing and Communications')
body(
    'In 2020, ViT [4] came to be and gave rise to a quick surge of Medical images '
    'analysis applications. TransMed [17] adapted Multi-sequence MRI classification '
    'using cross-modal attention. This study employed another architectural strategy, '
    'the Swin Transformer [7]. — hierarchical shifted windows which create a '
    'spatial pyramid. Within one model — and particularly on the DPD model. '
    'tasks that must be performed simultaneously with sensitivity to local texture. '
    'global structure. The dual-resolution ViT encoders are applied to Handling the '
    'microscopic image of fungi and examination of their predictions. This has not '
    'been reported before with the exception of the case where should be fused.'
)
add_subsection_heading('Multi-Scale Fusion and Confidence Weighting')
body(
    'There is a vast amount of experience with multi-scale strategies in object '
    'detec- Most of the fusion work has been done through feature pyramid networks '
    '[11], but not this. in classification does not work at the prediction level, '
    'but rather at the feature level. level. Probability-level ensembling is less '
    'complicated and does not need any between encoders, and — importantly '
    '— admits non-parametric confidence weighting as in attention-, a random '
    'variable that is independent of the outcome. are determined by '
    'multiple-instance learning [12] are sampled from the model’s prediction '
    'distribution. Our The idea is extended to the multi-resolution setting with '
    'S3 strategy.'
)
add_subsection_heading('Clinical Models Explainability')
body(
    'Since then, Grad-CAM [9] has been the most popular attribution method for '
    'Medical imaging CNNs. The situation is the same for transformers. less settled. '
    'There is a principled alternative, Attention Rollout [8]. architecture that '
    'allows it to propagate the multi-head attention to each layer that Spatial '
    'coherent attribution through residual correction. Building maps without the '
    'need to perform backpropagation. That gradient-free The practicality of '
    'property makes it applicable in inference-only deployments, and This is '
    'frequently the case with clinical software.'
)

# ── SECTION III: DATASET ─────────────────────────────────────────────────────
add_section_heading('Dataset and Preprocessing')
add_subsection_heading('DeFungi Benchmark')
body(
    'DeFungi [10] is a publicly available collection of bright- The author '
    'discusses the application of microscopy in the field in order to obtain '
    'microscopic images from direct mycological exams. tions, including five '
    'species. We use an official split through- out: 5,000 training, 899 '
    'validation, and 902 test images. Table I Analyzes the distribution of '
    'classes. Two things are worth noticing: There are two items to note: noting: '
    'C. albicans (H1) accounts for 40% of the training samples and In contrast, '
    'E. floccosum (H6) only accounts for 10%, so this results in a A moderate '
    'imbalance that must be made up for by augmentation. The size of the validation '
    'and test splits is smaller compared to the size of the training set for a few '
    'classes (H3, H5 and H6) the total number of students is smaller. A few '
    'specimens, that still exist in the original collection.'
)

# Table I
tbl_cap1 = doc.add_paragraph()
tbl_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tbl_cap1.add_run('TABLE I.    DEFUNGI DATASET CLASS DISTRIBUTION')
set_font(r, size=9, bold=True)
set_para_spacing(tbl_cap1, before=60, after=20)

t1 = doc.add_table(rows=7, cols=5)
t1.style = 'Table Grid'
t1.alignment = WD_TABLE_ALIGNMENT.CENTER
headers1 = ['ID', 'Species', 'Train', 'Val', 'Test']
data1 = [
    ('H1', 'C. albicans',       '2,000', '437', '437'),
    ('H2', 'A. niger',          '750',   '233', '233'),
    ('H3', 'T. rubrum',         '750',   '82',  '82'),
    ('H5', 'T. mentagrophytes', '1,000', '80',  '80'),
    ('H6', 'E. floccosum',      '500',   '67',  '70'),
    ('',   'Total',             '5,000', '899', '902'),
]
for j, h in enumerate(headers1):
    cell = t1.rows[0].cells[j]
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cell.paragraphs[0].add_run(h)
    set_font(r, size=9, bold=True)
for i, row_data in enumerate(data1):
    for j, val in enumerate(row_data):
        cell = t1.rows[i+1].cells[j]
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        bold = (i == 5)
        italic = (j == 1 and i < 5)
        r = cell.paragraphs[0].add_run(val)
        set_font(r, size=9, bold=bold, italic=italic)

add_subsection_heading('Augmentation Pipeline')
body(
    'For training images, the images are randomly sized and cropped. and '
    'horizontal and vertical flipping (scale 0.8–1.0), Brightness, contrast '
    'and saturation each ±0.4 (colour jitter). Ran- editorial flexibility in '
    'erasing the text. It gives some editorial freedom to have text erased [15] '
    'with a 25% chance. The occlusion robustness is important for specimens where '
    'The field of view contains debris blocking part of the view. At batch level '
    'we apply MixUp [13] (α = 0.8) and CutMix [14] both strategies soften '
    'the (α = 1.0) with equal probability; both strategies soften the '
    'inter-class boundaries and have helped in other fine-grained Limited data for '
    'classification settings. Validation and test It is known that images only see '
    'centre-crop normalisation (IN) with ImageNet. statistics. All the inputs are '
    'independently resized to each encoder at its own target resolution (224 or '
    '384 pixels), anti- aliased bilinear interpolation.'
)

# ── SECTION IV: METHODOLOGY ───────────────────────────────────────────────────
add_section_heading('Methodology')
add_subsection_heading('Backbone Encoders')
body(
    'The weights of both encoders are initialized with weights from ViT-B/16 [4] '
    'pretrained. on timm loaded ImageNet-21k. The original classification head is '
    'swapped for a randomly initialised linear The [CLS] token is mapped to 5 with '
    'the following mapping: logits. All 12 of the transformer blocks are trained '
    'from the first We don’t freeze any layers, you will work through the '
    'entire period in one sitting.'
)
body(
    'The two encoders are the same, except for the number of inputs. ViT-224 '
    'splits each image into 14 × 14 = 196 non-overlapping 16 × 16 '
    'patches. ViT-384 produces 24 × 24 = 576 patches, providing almost three '
    'times the space tokens and hence A finer resolution of the image. The timm '
    'ViT-B/16- Positional embeddings are shared between the 224- and 384-variant '
    'variants. Positional embeddings are identical for both the 224- and '
    '384-variant variants. This method does not use bicubic interpolation, so it '
    'is referred to as pixel checkpoint. instability in training that can occur '
    'when positional encodings Randomly initialise are at new resolution.'
)

add_subsection_heading('ACMF Fusion Framework')
body(
    'Let p₂₂₄ and p₃₈₄ be the softmax output '
    'vectors from For a particular image, both ViT-224 and ViT-384 have a value '
    'of Δ^(C−1). C = 5. The four strategies that are assessed are '
    'detailed below.',
    first_indent=True
)

p_s1 = doc.add_paragraph()
p_s1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r1 = p_s1.add_run('S1 — Equal-Weight Averaging: ')
set_font(r1, size=10, bold=True)
r1b = p_s1.add_run(
    'No assumptions about which encoder is better; the simplest possible baseline '
    'for fusion.'
)
set_font(r1b, size=10)
set_para_spacing(p_s1, before=20, after=20)
equation_para('1', 'ᵖ̂⁽¹⁾ = ½ p₂₂₄ + ½ p₃₈₄')

p_s2 = doc.add_paragraph()
p_s2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r2 = p_s2.add_run('S2 — Accuracy-Weighted Fusion: ')
set_font(r2, size=10, bold=True)
r2b = p_s2.add_run(
    'wᵣ = aᵣ / (a₂₂₄ + a₃₈₄), '
    'r ∈ {224, 384}, where aᵣ is the validation-set top-1 accuracy of '
    'encoder r. The fused output becomes '
    'ᵖ̂⁽²⁾ = w₂₂₄ p₂₂₄ + '
    'w₃₈₄ p₃₈₄. This places more trust in the encoder '
    'that generalised better, but the weights are fixed across all test images.'
)
set_font(r2b, size=10)
set_para_spacing(p_s2, before=20, after=20)

p_s3 = doc.add_paragraph()
p_s3.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r3 = p_s3.add_run('S3 — Confidence-Guided Entropy Weighting: ')
set_font(r3, size=10, bold=True)
r3b = p_s3.add_run(
    'Rather than assigning the same weight to every image, S3 computes per-sample '
    'weights based on how confident each encoder is. We measure confidence through '
    'Shannon entropy H(p) = −Σᴄ pᴄ log pᴄ: lower entropy '
    'means the model is more certain about its prediction. For each image i: '
    'wᵣ⁽⁩⁾ = (H(pᵣ⁽⁩⁾) + ε)⁻¹ '
    '/ Σᵣ′(H(pᵣ′⁽⁩⁾) + ε)⁻¹, '
    'ε = 10⁻⁸. The fused prediction is '
    'ᵖ̂⁽³⁾ᵢ = w₂₂₄⁽⁩⁾ '
    'p₂₂₄⁽⁩⁾ + w₃₈₄⁽⁩⁾ '
    'p₃₈₄⁽⁩⁾. '
    'The pair (w₂₂₄⁽⁩⁾, w₃₈₄⁽⁩⁾) '
    'is an interpretable artifact of inference: it tells us which resolution the '
    'model found more decisive for specimen i, at no extra computational cost '
    'beyond computing entropy of an already-available softmax vector.'
)
set_font(r3b, size=10)
set_para_spacing(p_s3, before=20, after=20)

p_s4 = doc.add_paragraph()
p_s4.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r4 = p_s4.add_run('S4 — Learnable MLP Fusion Head: ')
set_font(r4, size=10, bold=True)
r4b = p_s4.add_run(
    'A two-layer MLP with LayerNorm and ReLU takes the concatenated vector '
    '[p₂₂₄; p₃₈₄] ∈ ℝ¹⁰ and '
    'outputs two softmax-normalised mixture weights. The head is trained for 20 '
    'epochs on frozen backbone outputs using AdamW with cosine annealing, so that '
    'the weighting function can in principle learn arbitrary, specimen-conditioned '
    'asymmetries.'
)
set_font(r4b, size=10)
set_para_spacing(p_s4, before=20, after=20)

add_subsection_heading('Training Protocol')
body(
    'Training uses AdamW, starting at a learning rate of 10⁻⁴ Weight '
    'decay set to 10⁻². After that, the learning rate changes along a '
    'cosine schedule A rule follows a straight 5-epoch ramp-up phase. With label '
    'smoothing applied at ε = 0.1 Instead of rigid training goals, [16] '
    'introduces flexible alternatives. During the learning process, these '
    'adjustments occur consistently use automatic mixed precision '
    '(torch.amp.GradScaler) Using less memory helps ease pressure on the RTX '
    '4050’s limited 6 GB VRAM Yet maintaining numerical stability throughout. '
    'The top checkpoint F1 macro preserves it, while early stopping activates '
    'should After ten straight rounds without a gain in validation F1, progress '
    'stalls Thirty epochs set as upper limit; convergence of ViT-224 reached by '
    'that point Twenty-eight, along with ViT-384, applied every one of the thirty.'
)

add_subsection_heading('Attention Rollout for Explainability')
body(
    'Attention Generated After Every ViT Forward Pass Rollout map [8]. The method '
    'multiplies the raw attention Through each of the twelve layers one after '
    'another, using a matrix residual correction at each step: '
    'Â_l = 0.5 A_l + 0.5 I. The product ∏_l Â_l gives per-patch '
    'attribution scores relative to Starting with the [CLS] token, elements are '
    'restructured into the layout of the patch grid Starting at higher detail, '
    'resolution matches the original input. This produces a map showing warmth '
    'levels across areas overlay on the original slide image, giving a reviewer a '
    'direct A close connection appears between what the model forecasts and the '
    'tiny details seen under magnification Its foundation depended on certain '
    'frameworks.'
)

# ── SECTION V: RESULTS ────────────────────────────────────────────────────────
add_section_heading('Tests With Findings')
body(
    'Each test takes place using one NVIDIA RTX 4050. Running everything on just '
    'that hardware keeps conditions steady. The setup does not include extra chips '
    'or added processors. Performance data comes solely from this card. No '
    'alternate configurations appear in the trials. Testing stays limited to this '
    'specific model throughout Laptop GPU featuring 6 GB of VRAM, compatible with '
    'CUDA version 12.9 alongside PyTorch release 2.1 Top-1 accuracy appears '
    'alongside macro-averaged F1, known as F1-M, then weighted metrics follow '
    'closely behind With regard to the held-out set, performance is assessed via '
    'F1 for class W, along with macro-averaged precision and individual class-level '
    'F1 scores Among 902 images used for testing, inference latency reflects '
    'average GPU processing duration Across one thousand attempts, performance '
    'followed fifty initial adjustment steps. Each run built on prior iterations '
    'without abrupt shifts. Progress emerged gradually through repeated exposure. '
    'Stability came only after early fluctuations settled down. A single frame '
    'guided every sequence throughout testing Higher performance emerges through '
    'adjustments to ResNet-50, though improvements also appear when modifying '
    'EfficientNet-B3; similarly, Swin-Tiny shows gains after refinement Adjusted '
    'within DeFungi using the identical setup applied to the ViT encoders Besides '
    'functioning as starting points, they provide reference markers.'
)

add_subsection_heading('Performance of Single Models')
body(
    'Results appear in Table II. Leading performance emerges from Swin-Tiny With '
    'a precision of 91.02%, alongside an F1 macro score reaching 92.66%. Our view '
    'aligns here Its four-level structure manages tasks well due to clear '
    'progression through stages Across one forward pass, different spatial '
    'resolutions emerge — some One goal of the dual-encoder setup is to '
    'mirror a specific pattern it learns during training A different approach '
    'appears here. One system trails slightly behind, scoring 91.88% on F1 macro. '
    'This version follows ViT-384 in rank It becomes clear the added detail '
    'reveals structural form more fully Details lost when scaled down to 224 pixels.'
)

# Table II
tbl_cap2 = doc.add_paragraph()
tbl_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tbl_cap2.add_run('TABLE II.    INDIVIDUAL MODEL PERFORMANCE ON DEFUNGI TEST SET')
set_font(r, size=9, bold=True)
set_para_spacing(tbl_cap2, before=60, after=20)

t2 = doc.add_table(rows=6, cols=5)
t2.style = 'Table Grid'
t2.alignment = WD_TABLE_ALIGNMENT.CENTER
h2 = ['Model', 'Acc.(%)', 'F1-M(%)', 'Lat.(ms)', 'Params(M)']
d2 = [
    ('ResNet-50',       '78.71', '81.61', '5.8',   '23.5'),
    ('EfficientNet-B3', '84.26', '88.04', '12.7',  '10.7'),
    ('ViT-B/16-224',    '84.70', '88.47', '8.5',   '85.8'),
    ('Swin-Tiny',       '91.02', '92.66', '11.8',  '27.5'),
    ('ViT-B/16-384',    '89.69', '91.88', '101.5', '86.1'),
]
for j, h in enumerate(h2):
    c = t2.rows[0].cells[j]
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.paragraphs[0].add_run(h)
    set_font(r, size=9, bold=True)
for i, rd in enumerate(d2):
    for j, val in enumerate(rd):
        c = t2.rows[i+1].cells[j]
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        bold = (i == 3 and j in [1,2])
        r = c.paragraphs[0].add_run(val)
        set_font(r, size=9, bold=bold)

body(
    'Close behind, ViT-224 performs about the same as EfficientNet-B3. Their '
    'results sit nearly side by side Though ResNet-50 scores lower at 81.61%, '
    'performance reaches nearly 88.5% F1 in contrast ResNet’s gap probably '
    'stems from missing long-range connections entirely note well: several unique '
    'traits appear throughout this collection — Shape across conidial '
    'structures, alongside patterns in branching complexity Arranged in layers, '
    'topology shows properties that span beyond single points; consider a sequence '
    'of 3 by 3 units stacked together Because convolutions process step by step, '
    'they miss simultaneous patterns. This delay A different perspective emerges '
    'clearly: ViT-384 processes each image in 101.5 milliseconds At 8.5 ms, '
    'twelve times below ViT-224’s speed, output drops sharply Close to ten '
    'pictures every second. This limitation affects everything A system designed '
    'to handle numerous samples at once begins operation. While one step finishes, '
    'another loads into place. After preparation comes analysis, followed by '
    'storage of results. Each specimen moves through stages without delay. Before '
    'the cycle ends, data gets reviewed automatically. Once complete, the next '
    'batch enters seamlessly clinic session.'
)

add_subsection_heading('ACMF Fusion Strategies')
body(
    'Results from Table III alongside Figure 1 reveal patterns across the four '
    'fusion methods When compared, S2 — adjusted for accuracy — shows '
    'the highest F1 macro score standing at 91.63%, a figure that exceeds ViT-224 '
    'by 3.16 points yet only inches ahead by 0.25 points Higher than ViT-384 when '
    'combining both encoders. S1 with S3 followed by S4 at 91.27%, while the '
    'prior group holds steady near 91.6%. Though slightly lower, both remain '
    'within a narrow margin overall When the MLP is trained, performance drops '
    'again to 88.59%.'
)

# Table III
tbl_cap3 = doc.add_paragraph()
tbl_cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tbl_cap3.add_run('TABLE III.    ACMF FUSION STRATEGY COMPARISON (DEFUNGI TEST SET)')
set_font(r, size=9, bold=True)
set_para_spacing(tbl_cap3, before=60, after=20)

t3 = doc.add_table(rows=7, cols=5)
t3.style = 'Table Grid'
t3.alignment = WD_TABLE_ALIGNMENT.CENTER
h3 = ['Strategy', 'Acc.(%)', 'F1-M(%)', 'F1-W(%)', 'Prec-M(%)']
d3 = [
    ('S1 Equal Weight',      '89.47', '91.59', '89.50', '90.86'),
    ('S2 Accuracy Weighted', '89.36', '91.63', '89.39', '90.80'),
    ('S3 Confidence Guided', '89.25', '91.35', '89.29', '90.71'),
    ('S4 Learnable MLP',     '84.92', '88.59', '85.07', '87.96'),
    ('ViT-224 (ref.)',        '84.70', '88.47', '84.86', '87.84'),
    ('ViT-384 (ref.)',        '89.69', '91.88', '89.70', '91.41'),
]
for j, h in enumerate(h3):
    c = t3.rows[0].cells[j]
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.paragraphs[0].add_run(h)
    set_font(r, size=9, bold=True)
for i, rd in enumerate(d3):
    for j, val in enumerate(rd):
        c = t3.rows[i+1].cells[j]
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        bold = (i == 1 and j == 2)
        r = c.paragraphs[0].add_run(val)
        set_font(r, size=9, bold=bold)

# Figure placeholder
fig_para = doc.add_paragraph()
fig_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fig_para.add_run('[Figure 1: ACMF Fusion Strategy Comparison Bar Chart]')
set_font(r, size=9, italic=True)
set_para_spacing(fig_para, before=60, after=20)

fig_cap = doc.add_paragraph()
fig_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fig_cap.add_run(
    'Fig. 1.  F1 macro and accuracy scores for all four ACMF fusion strategies, '
    'each shown in a distinct colour.'
)
set_font(r, size=9)
set_para_spacing(fig_cap, before=0, after=60)

body(
    'The narrow gap between S1 S2 and S3 just 0.28 points Ranking outcomes '
    '(starting at the lowest) offers insight on its own. This implies that the '
    'primary benefit of the ACMF framework comes from Putting together two '
    'resolutions entirely, though never due to the specific mech — One method '
    'applies a system to assign value. These approaches rely mostly on practical '
    'rules Though differing slightly, they perform similarly on macro F1, a sign '
    'that improvements hold steady even when conditions shift. That consistency '
    'suggests strength in design, not just fine-tuned luck Adjusting the parameters '
    'within the weight assignment method.'
)
body(
    'Looking at S4’s weaker results calls for a closer look. That pattern '
    'stands out when compared to the MLP Most expressive by nature, the head moves '
    'with subtle shifts. Yet its range depends on context alone. Without gesture '
    'support, impact fades quickly. Expression grows stronger when paired with '
    'posture cues. Still, limitations appear under stress. In calm moments, clarity '
    'rises easily Though limited to just 5,000 training samples, the model still '
    'uses a dual-encoder setup with softmax scoring. Each encoder processes inputs '
    'separately before comparing outputs through probability-based alignment A '
    'point moves through ten dimensions where shape shifts occur unpredictably. '
    'Structure transforms as coordinates evolve across layered directions. '
    'Dimensions reconfigure while vectors follow new alignments. Direction alters '
    'when spatial rules change without warning. Movement continues even as '
    'framework bends into unfamiliar forms Some encoders might differ slightly '
    'based solely on their initial calibration moment Optimising this process '
    'proves challenging due to its complexity. Despite the difficulties, progress '
    'emerges slowly through trial. Small adjustments often lead to better outcomes '
    'over time. Each tweak builds on earlier attempts without guaranteeing success. '
    'The path forward remains uncertain at every step A handful of measurements. '
    'Despite limited information, S3 uses entropy as a guiding principle wholly '
    '— it functions without instruction, avoiding assumptions about data '
    'patterns yet still matches the performance of S1 alongside S2 Though '
    'producing a clear confidence measure for each image as a byproduct.'
)
body(
    'When it comes to how clearly S3 can be understood, examining the recorded '
    'weight pairings offers insight (w₂₂₄⁽⁩⁾, '
    'w₃₈₄⁽⁩⁾) Throughout the evaluation data, it '
    'became clear that ViT-224 showed consistent patterns Got consistently greater '
    'importance — less uncertainty — in H5 H6 samples show traits best '
    'seen at larger scales Appearing clearly under regular magnification, its form '
    'stands out without needing enhanced detail — ViT-384 Throughout H2 and '
    'H3, clear dominance appears through delicate hyphal wall structures. Fine '
    'details emerge under close observation. These features stand out distinctly '
    'within both sections. Subtle textures reveal themselves gradually. Structure '
    'clarity improves at higher magnification levels What sets them apart lies in '
    'how far apart the septa are placed. That detail alone makes the difference '
    'clear It might seem obvious, yet watching how the model acts matches what '
    'you’d think Deep within fungal patterns, insight emerges not through '
    'proof but recognition. What feels right often bypasses logic entirely. '
    'Patterns click when least expected. A sense unfolds slowly, shaped by repeated '
    'observation. Knowing arrives without explanation sometimes. Experience feeds '
    'quiet certainty over time The peak performance metric in classification tasks.'
)

add_subsection_heading('Per-Class Analysis')
body(
    'Looking at Table IV, outcomes are sorted by class. This reveals a consistent '
    'trend Across every model, one pattern stands out — H2 (A. niger) '
    'consistently ranks as the most difficult category H5 along with H6 stand out '
    'as simplest. At 60.09%, ResNet-50’s H2 reaches an F1 score stays notably '
    'low — about 25 percent of H2 samples fall into this range Some get '
    'overlooked entirely. Others mix up identities across similar kinds. Focus '
    'matters — yet slips often Here, models built on newer architectures '
    'perform significantly stronger — ViT-384 hits 82.80%. While older '
    'designs lag behind, this version shows what’s possible through refined '
    'attention layers. Its accuracy stands out clearly when tested under identical '
    'conditions Global context appears important, given Swin-Tiny reached 85.22% '
    'For identifying A. niger, progress has been notable. Yet challenges remain '
    'present surprising; A. niger specimens in this dataset show substantial '
    'Conidial head dimensions shift slightly when branch angles change. Sometimes, '
    'larger structures appear alongside narrower forks. Branching patterns differ '
    'across specimens. Size alters as angles widen or tighten. Differences emerge '
    'under close examination. Shape adjustments occur with angular shifts. '
    'Dimensions respond to structural changes Depending on how they grow, these '
    'factors become more elusive to identify compared to Four additional types '
    'exist apart from that group.'
)

# Table IV
tbl_cap4 = doc.add_paragraph()
tbl_cap4.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tbl_cap4.add_run('TABLE IV.    PER-CLASS F1 SCORE (%) — DEFUNGI TEST SET')
set_font(r, size=9, bold=True)
set_para_spacing(tbl_cap4, before=60, after=20)

t4 = doc.add_table(rows=7, cols=7)
t4.style = 'Table Grid'
t4.alignment = WD_TABLE_ALIGNMENT.CENTER
h4 = ['Model', 'H1', 'H2', 'H3', 'H5', 'H6', 'F1-M']
d4 = [
    ('ResNet-50',       '82.96', '60.09', '79.78', '92.59', '92.65', '81.61'),
    ('EfficientNet-B3', '85.58', '73.49', '88.89', '94.41', '97.81', '88.04'),
    ('ViT-B/16-224',    '85.03', '75.60', '91.12', '95.65', '94.96', '88.47'),
    ('ViT-B/16-384',    '90.40', '82.80', '92.77', '96.25', '97.18', '91.88'),
    ('Swin-Tiny',       '91.82', '85.22', '93.75', '96.86', '95.65', '92.66'),
    ('S2 ACMF (ours)',  '90.06', '82.43', '93.41', '95.06', '97.18', '91.63'),
]
for j, h in enumerate(h4):
    c = t4.rows[0].cells[j]
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.paragraphs[0].add_run(h)
    set_font(r, size=9, bold=True)
for i, rd in enumerate(d4):
    for j, val in enumerate(rd):
        c = t4.rows[i+1].cells[j]
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        bold = (i == 5 and j == 6)
        r = c.paragraphs[0].add_run(val)
        set_font(r, size=9, bold=bold)

body(
    'Notably, the S2 fusion achieves 93.41% on H3 (T. rubrum), higher than each '
    'separate encoder at 92.77% and With a score of 91.12%, this instance stands '
    'out as a notably clear case of merging elements offering real improvement '
    'compared to each part alone. Regarding H2, S2 achieves 82.43%, a touch lower '
    'than ViT-384 when tested alone Despite outperforming ViT-224 by a notable '
    'margin, the overall impact of merging remains evident Good results came '
    'through for this group.'
)

add_subsection_heading('Explainability Results')
body(
    'Looking at attention rollout maps on the test set shows where correct '
    'Predictions link to where attention settles in space Living systems often '
    'involve patterns that matter. When studying H1 samples, focus shifts here '
    'Focus shifts to sequences of yeast cells forming buds; regarding H2 and also '
    'H3, On hyphal surfaces and at junctions where strands split, light settles; '
    'here, H5 meets H6 Focus shifts toward the pointed ends of macroconidia along '
    'with club-shaped terminal cells For H6, a notable point emerges about correct '
    'outcomes in order Among classified samples, rollout mapping typically gives '
    'almost negligible weight Background hyphae gain emphasis here, shifting '
    'attention nearly entirely away from other elements At the tip, where it '
    'thickens slightly — that particular shape defines the form A mycologist '
    'might search for marker a during their examination.'
)
body(
    'Looking at wrong labels reveals something else entirely. Where the focus '
    'actually lands becomes clear only through these errors Most maps showing '
    'wrongly labeled H2 samples appear nearly identical across the board Across '
    'the picture, details blur into one another. Blended throughout, features lose '
    'clear borders. Without distinct form, patterns drift apart. Scattered widely, '
    'elements avoid fixed positions. Flowing freely, shapes mix in quiet chaos. '
    'Every part shares a soft, even presence True. Perhaps these samples really do '
    'not have the Features of shape and structure noticed by the system — '
    'maybe also a person Someone skilled might turn to this instead. In real '
    'situations, it may help when applied carefully Some cases show strong softmax '
    'signals alongside scattered spatial patterns attention may warrant flagging '
    'for expert review even when The model shows clear confidence, given its '
    'suggestion of a specific choice fails to rely on the typical distinguishing '
    'characteristics.'
)

# ── SECTION VI: DISCUSSION ────────────────────────────────────────────────────
add_section_heading('Discussion')
body(
    'The result that stands out most is Swin-Tiny outperforming all four ACMF '
    'strategies despite using a single model and A single run through the model. '
    'Performance reaches 92.66% on F1 macro, slightly above the prior 91.63% '
    'Though small, the edge exists for optimal integration methods Across every '
    'measure used to assess performance, results remained stable. Our view is that '
    'the explanation Architecture shapes the nation: Swin’s method uses '
    'layered window attention A structure forms inside, creating a layered system '
    'across different sizes At the same time, ACMF handles it through execution '
    'One encoder operates at a set resolution; the other does likewise before '
    'integration occurs. Their outputs merge through alignment mechanisms rather '
    'than simple addition Results appear as likelihood values. How the system '
    'stores information inside contains more information than what emerges through '
    'after-the-fact likelihood analysis weighting.'
)
body(
    'Still, it might be wise to avoid interpreting this as proof It turns out ACMF '
    'lacks practical value. Although built with intent, the structure fails under '
    'real conditions because it does not adapt well. Its original purpose becomes '
    'unclear once applied outside theory. Problems emerge quickly when tested in '
    'live scenarios where flexibility matters most Working across multiple '
    'resolutions helps performance compared to using just one resolution, especially '
    'beyond a basic ViT setup where only a fixed scale is applied Where advanced '
    'design exists but remains unused, or proves impractical, Clearly, it operates '
    'without ambiguity. In a given setting, consistency remains evident Demands for '
    'ViT emerge — maybe due to established systems, Access to pretrained '
    'models, alongside how clearly they explain their decisions Attention Rollout’s '
    'method — combining dual resolutions through fusion — is one approach '
    'A shift that works within existing structures. Still, it brings progress '
    'without demanding redesign.'
)
body(
    'The comparison among S1 S2 S3 and S4 reveals patterns A helpful tale '
    'describing the sort of blending needed for this job. What follows is similar '
    'performance among three heuristic approaches indicates their comparable '
    'effectiveness Most of the time, handling the task doesn’t require '
    'detailed adjustments for each individual image This amount of data. When S4 '
    'falls short, it echoes a pattern seen before lesson: learned ensemble '
    'weighting is more data-hungry than A single model alone might seem too '
    'simplistic. Yet five thousand pictures during learning still fall short So '
    'that the MLP can extend past the trained encoders, calibration patterns'
)

# ── SECTION VII: CONCLUSION ───────────────────────────────────────────────────
add_section_heading('Conclusion')
body(
    'The ACMF framework was presented, followed by an assessment of four strategies '
    'for combining ViT-B/16 predictions at two input resolutions on the DeFungi '
    'microscopic fungal classification On average, the method that balances '
    'precision and correctness (S2) reaches F1 macro stands at 91.63%, rising by '
    '3.16 points compared to ViT-224 Just one, sitting at a mere quarter-point '
    'beneath the ViT-384 that runs twelve times slower Approach. The method using '
    'entropy guidance (S3) achieves similar performance Achieving performance '
    'levels alongside generating confidence indicators for each image which '
    'indicate to an operator the resolution selected by the model One sample at a '
    'time, the process moves forward. Across the evaluation group, rollout patterns '
    'appear clearly Checking whether the model’s forecasts tie back to '
    'identical foundations Features of shape and form studied by someone who '
    'researches fungi, What counts toward trust in medical settings.'
)
body(
    'It is clear that constraints exist. One must consider the impact of operating '
    'dual encoders inference speed drops by about half when compared to one model '
    'alone Offline or batch settings become the main scenarios, given how '
    'limitations shape real-world application Yet performance dips slightly below '
    'Swin-Tiny despite minimal computational cost Internally managing several '
    'scales adjusts the number of parameters involved. Because of limited data, '
    'the trained S4 head faces restrictions in its performance Heuristic methods '
    'function differently compared to these approaches.'
)
body(
    'Future possibilities appear most compelling when combining Swin-based encoder '
    'with a ViT encoder rather than two ViTs Because of varied building styles, '
    'resilience might improve overall What matters most is how pieces fit together, '
    'not just clarity gaps. Insight distillation from the dual-encoder pipeline '
    'into a lightweight A lone framework might still hold value, since it may '
    'uncover gains in performance come without added delay. Lastly, the Even when '
    'models grow smarter, data size still holds things back — using unlabeled '
    'examples helps ease the load pretraining on unlabelled clinical microscopy '
    'data could ex- Yet still expanding the useful examples for learning, though '
    'no extra data gets added expert annotation.'
)

# ── ACKNOWLEDGMENT ────────────────────────────────────────────────────────────
ack_head = doc.add_paragraph()
ack_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = ack_head.add_run('ACKNOWLEDGMENT')
set_font(r, size=10)
set_para_spacing(ack_head, before=80, after=40)

body(
    'The Authors Thank The Department Of Computer Science And Engineering at '
    'Karunya Institute of Technology and Sciences Computational tools were made '
    'available through support from. [Include acknowledgment of financial backing '
    'if applicable.]',
    first_indent=False
)

# ── REFERENCES ────────────────────────────────────────────────────────────────
ref_head = doc.add_paragraph()
ref_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = ref_head.add_run('REFERENCES')
set_font(r, size=10)
set_para_spacing(ref_head, before=80, after=40)

refs = [
    '[1] K. Kainz, M. Lakemeyer, and F. Madeo, "Fungal diseases — forgotten killers? Understanding the global burden of fungal infections," EMBO Mol. Med., vol. 12, no. 8, p. e12114, 2020.',
    '[2] P. E. Verweij and J. F. Meis, "Microbiological diagnosis of invasive fungal infections: challenges and progress," J. Antimicrob. Chemother., vol. 55, suppl. 1, pp. 15–22, 2005.',
    '[3] C. L. Srinidhi, O. Ciga, and A. L. Martel, "Deep neural network models for computational histopathology: a survey," Med. Image Anal., vol. 67, p. 101813, 2021.',
    '[4] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in Proc. ICLR, 2021.',
    '[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE CVPR, pp. 770–778, 2016.',
    '[6] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in Proc. ICML, vol. 97, pp. 6105–6114, 2019.',
    '[7] Z. Liu et al., "Swin transformer: Hierarchical vision transformer using shifted windows," in Proc. IEEE ICCV, pp. 10012–10022, 2021.',
    '[8] S. Abnar and W. Zuidema, "Quantifying attention flow in transformers," in Proc. ACL, pp. 4190–4197, 2020.',
    '[9] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," Int. J. Comput. Vis., vol. 128, no. 2, pp. 336–359, 2020.',
    '[10] P. Cassagne et al., "DeFungi: Direct mycological examination of microscopic fungi images," arXiv preprint arXiv:2110.04984, 2021.',
    '[11] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature pyramid networks for object detection," in Proc. IEEE CVPR, pp. 2117–2125, 2017.',
    '[12] M. Ilse, J. M. Tomczak, and M. Welling, "Attention-based deep multiple instance learning," in Proc. ICML, vol. 80, pp. 2127–2136, 2018.',
    '[13] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "MixUp: Beyond empirical risk minimization," in Proc. ICLR, 2018.',
    '[14] S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo, "CutMix: Training strategy that makes strongly localized contributions matter," in Proc. IEEE ICCV, pp. 6023–6032, 2019.',
    '[15] Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, "Random erasing data augmentation," in Proc. AAAI, vol. 34, pp. 13001–13008, 2020.',
    '[16] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the Inception architecture for computer vision," in Proc. IEEE CVPR, pp. 2818–2826, 2016.',
    '[17] Y. Dai et al., "TransMed: Transformers advance multi-modal medical image classification," Diagnostics, vol. 11, no. 8, p. 1384, 2021.',
]

for ref_text in refs:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = para.add_run(ref_text)
    set_font(r, size=9)
    set_para_spacing(para, before=0, after=20)
    pPr = para._p.get_or_add_pPr()
    ind = OxmlElement('w:ind')
    ind.set(qn('w:left'),      '360')
    ind.set(qn('w:hanging'),   '360')
    pPr.append(ind)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = r'D:\Fungi\fungi-ai\paper\manuscript.docx'
doc.save(out_path)
print(f'Saved: {out_path}')
