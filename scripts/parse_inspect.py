#!/usr/bin/env python3
"""Tong hop cac dong '[inspect]' tu log cua `--inspect`.

Dung:  python scripts/parse_inspect.py log_a.txt log_b.txt ...

Moi cua so sinh ra mot khoi gom cac bien the:
  base           -- neo tren mo hinh DA TROI, gradient that  (= cach goc chay)
  anchor_global  -- neo tren mo hinh GLOBAL,  gradient that
  shuffle        -- hoan vi toa do lop (giu do lon, pha canh nhan)
  foreign        -- gradient cua cua so dau tien (client khac)
  scale0.1/10    -- gradient nhan ti le

Doc ket qua:
  base - anchor_global  lon  => thong tin cua neo den tu TRONG SO GHI NHO
                               phan phoi client trong m buoc local.
  base - shuffle        nho  => gradient khong thuc su duoc doc.
  base vs scale*        lech => uoc luong KHONG bat bien thang do.
"""
import re
import sys
from collections import defaultdict

ORDER = ['base', 'anchor_global', 'shuffle', 'foreign', 'scale0.1', 'scale10']
METHODS = ['RLU', 'ZLGp', 'LLGp']


def parse(path):
    """Tra ve list cac dict {bien_the: {pp: irec}}, moi dict la mot cua so."""
    windows, cur = [], {}
    for ln in open(path, encoding='utf-8', errors='replace'):
        m = re.search(r'\[inspect\]\s+(\S+)\s+(.*)', ln)
        if not m:
            continue
        variant = m.group(1)
        if variant in cur:          # gap lai 'base' => cua so moi
            windows.append(cur)
            cur = {}
        vals = {k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', m.group(2))}
        if vals:
            cur[variant] = vals
    if cur:
        windows.append(cur)
    return windows


def main(paths):
    for path in paths:
        wins = parse(path)
        if not wins:
            print('=== %s: khong tim thay dong [inspect] nao ===\n' % path)
            continue
        agg = defaultdict(lambda: defaultdict(list))
        for w in wins:
            for variant, vals in w.items():
                for meth, v in vals.items():
                    agg[variant][meth].append(v)

        print('=== %s (%d cua so) ===' % (path, len(wins)))
        print('  %-14s %s' % ('bien the', '  '.join('%-8s' % m for m in METHODS)))
        for variant in ORDER:
            if variant not in agg:
                continue
            row = []
            for m in METHODS:
                v = agg[variant].get(m, [])
                row.append('%-8s' % ('%.4f' % (sum(v) / len(v)) if v else '-'))
            note = ''
            if variant == 'foreign':
                note = '   (bo qua cua so dau: no tu lam "foreign" cua chinh no)'
            print('  %-14s %s%s' % (variant, '  '.join(row), note))

        print('  %s' % ('-' * 46))
        base = agg.get('base', {})
        for label, variant in [('neo ghi nho  (base - anchor_global)', 'anchor_global'),
                               ('gradient gop (base - shuffle)      ', 'shuffle'),
                               ('gradient gop (base - foreign)      ', 'foreign')]:
            if variant not in agg:
                continue
            row = []
            for m in METHODS:
                a, b = base.get(m, []), agg[variant].get(m, [])
                n = min(len(a), len(b))
                row.append('%-8s' % ('%+.4f' % (sum(a[:n]) / n - sum(b[:n]) / n) if n else '-'))
            print('  %s %s' % (label, '  '.join(row)))
        print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
