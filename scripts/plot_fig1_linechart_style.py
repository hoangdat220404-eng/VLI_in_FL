#!/usr/bin/env python3
"""Figure 1 (a/b/c): irec vs. global round, untrained -> converged,
in the EXACT matplotlib style of figure/Line_chart.py (the user's own
reference script) -- same figsize, fontsize, grid, external legend,
spine width, box_aspect, and the commented-out "CHUA TRAINED DEN DA
TRAINED" Model-Accuracy-overlay convention from that same file, now
actually enabled. Three separate PNG files, one per dataset/model pair.
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(_HERE, 'rq2_parsed.json'), encoding='utf-8'))

# Same palette as Line_chart.py's own uncommented block (RLU/LLG/ZLGp/LLGp/
# VLI), extended with 2 more colors/markers for iLRG and the 2 extra ZLI
# variants Line_chart.py never had to distinguish.
METHOD_STYLE = {
    'RLU':       dict(label='RLU',          color='#0000ff', marker='o'),
    'iRLG':      dict(label='iLRG',         color='#b36302', marker='v'),
    'LLG':       dict(label='LLG',          color='#800020', marker='*'),
    'LLGp':      dict(label='LLGp',         color='#0b8000', marker='s'),
    'ZLGp':      dict(label='ZLGp',         color='#ffa500', marker='^'),
    'VLI*max':   dict(label='ZLI-max (Ours)',     color='#ff0000', marker='D'),
    'VLI*mean+': dict(label='ZLI-mean+ (Ours)',   color='#cc0066', marker='X'),
    'VLI*med+':  dict(label='ZLI-med+ (Ours)',    color='#9400d3', marker='P'),
}
ORDER = ['RLU', 'iRLG', 'LLG', 'LLGp', 'ZLGp', 'VLI*max', 'VLI*mean+', 'VLI*med+']


def series(fname, main_alpha):
    pts = {}
    for b in D[fname]:
        if str(b.get('alpha')) != str(main_alpha):
            continue
        ep = int(b['ckpt_epoch'])
        if ep in pts:
            continue
        pts[ep] = {'acc': float(b['ckpt_train_acc']),
                    'irec': {m: b['table'][m][1] * 100 for m in b['table']}}
    return sorted(pts.items())


panels = [
    ('MobileNetV2 - SVHN', '../Experiment_Result/VLIstar_Fig1a_MobileNetV2_SVHN.txt', 0.5, 'fig1a_svhn.png'),
    ('ResNet18 - CIFAR10', '../Experiment_Result/VLIstar_Fig1b_ResNet18_CIFAR10.txt', 0.5, 'fig1b_cifar10.png'),
    ('ResNet18 - CIFAR100', '../Experiment_Result/VLIstar_Fig1c_ResNet18_CIFAR100.txt', 0.1, 'fig1c_cifar100.png'),
]

for title, fname, alpha, outname in panels:
    data = series(fname, alpha)
    epochs = [p[0] for p in data]
    model_accuracy = [p[1]['acc'] for p in data]

    # 2. Khoi tao do thi (exactly as in Line_chart.py)
    fig, ax = plt.subplots(figsize=(7, 7))

    # 3. Ve cac duong iAcc cho cac phuong phap tan cong
    for m in ORDER:
        ys = [p[1]['irec'][m] for p in data]
        st = METHOD_STYLE[m]
        ax.plot(epochs, ys, label=st['label'], marker=st['marker'],
                 linewidth=3.5, markersize=10, color=st['color'])

    # 4. Ve duong Model Accuracy mau den (do day noi bat hon)
    ax.plot(epochs, model_accuracy, label='Model Accuracy', color='black',
             marker='H', linestyle=(0, (5, 3)), linewidth=4, markersize=10)

    # 5. Cau hinh cac thuoc tinh hien thi theo yeu cau
    ax.set_xlabel('Global Round', fontsize=20)
    ax.set_ylabel('iAcc (%)', fontsize=20)
    ax.set_title(title, fontsize=20)

    ax.set_ylim(0, 100)
    ax.set_xticks(epochs)
    ax.tick_params(axis='x', labelsize=16, rotation=45)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(True, linestyle='--', color='black', linewidth=1.2, alpha=0.6)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=13, borderaxespad=0.)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')

    ax.set_box_aspect(0.8)
    plt.tight_layout()
    out = os.path.join(_HERE, '..', 'figures', outname)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)
