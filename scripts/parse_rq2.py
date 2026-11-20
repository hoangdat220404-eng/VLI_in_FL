#!/usr/bin/env python3
"""Parse 'Namespace(...check_point_path=...)' + 'Average over N clients'
blocks out of the RQ2 log files, pairing each checkpoint with its final
per-method (cAcc, iAcc) table. Pure text parsing, no GPU/model needed.
"""
import re
import sys
import json

NS_RE = re.compile(r"check_point_path='([^']*)'")
LR_RE = re.compile(r"\blr=([0-9.eE+-]+)")
LE_RE = re.compile(r"local_epochs=(\d+)")
ALPHA_RE = re.compile(r"\balpha=([0-9.eE+-]+)(?:,|\))")
AVG_HEAD_RE = re.compile(r"Average over (\d+) clients")
ROW_RE = re.compile(r"^([A-Za-z*+_]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*$")
CKPT_ACC_RE = re.compile(r"_acc_([0-9.]+)_(\d+)\.pth$")
# Measured accuracy of the model actually under attack (printed unconditionally
# by main.py right before the attack loop) -- preferred over parsing the
# checkpoint filename, which is how the Fig1a/b/c epoch-0 off-by-one bug was
# introduced, and the only source of a train-accuracy number for the
# untrained/randomly-initialized run (empty --check_point_path).
ATK_ACC_RE = re.compile(r"training accuracy cua mo hinh bi tan cong: top1=([0-9.]+)")
# Explicit signal that this run did NOT load any checkpoint (fresh/random init):
# only trust check_point_path=='' for this -- the argparse default is a
# non-empty filename that is simply ignored when --load_ckpt is absent, so a
# command that omits --check_point_path still prints that stale default string.
LOAD_CKPT_RE = re.compile(r"\bload_ckpt=(True|False)")


def parse(path):
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    blocks = []
    cur_ckpt = None
    cur_lr = cur_le = cur_alpha = None
    cur_atk_acc = None
    cur_load_ckpt = None
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith('Namespace('):
            m = NS_RE.search(line)
            if m:
                cur_ckpt = m.group(1)
            m = LR_RE.search(line)
            if m:
                cur_lr = m.group(1)
            m = LE_RE.search(line)
            if m:
                cur_le = m.group(1)
            m = ALPHA_RE.search(line)
            if m:
                cur_alpha = m.group(1)
            m = LOAD_CKPT_RE.search(line)
            cur_load_ckpt = (m.group(1) == 'True') if m else None
            cur_atk_acc = None  # reset: belongs to the run started by this Namespace
        m = ATK_ACC_RE.search(line)
        if m:
            cur_atk_acc = m.group(1)
        m = AVG_HEAD_RE.match(line.strip())
        if m:
            n_clients = m.group(1)
            # next line is the header 'Method | cAcc | iAcc', then rows until blank/non-matching
            j = i + 2
            table = {}
            while j < n:
                row = lines[j].strip()
                rm = ROW_RE.match(row)
                if not rm:
                    break
                method, cacc, iacc = rm.groups()
                table[method] = (float(cacc), float(iacc))
                j += 1
            ckpt_acc = ckpt_epoch = None
            if cur_ckpt and cur_load_ckpt is not False:
                # Old-style logs never printed load_ckpt at all (cur_load_ckpt
                # stays None) -- keep trusting the filename for those, exactly
                # as before. Only an *explicit* load_ckpt=False (new logs)
                # means no checkpoint was loaded this run.
                cm = CKPT_ACC_RE.search(cur_ckpt)
                if cm:
                    ckpt_acc, ckpt_epoch = cm.group(1), cm.group(2)
            if cur_atk_acc is not None:
                ckpt_acc = cur_atk_acc          # measured value wins over the filename
            if cur_load_ckpt is False:
                ckpt_epoch = '0'                 # untrained / randomly-initialized run
            blocks.append({
                'checkpoint': cur_ckpt,
                'ckpt_train_acc': ckpt_acc,
                'ckpt_epoch': ckpt_epoch,
                'lr': cur_lr,
                'local_epochs': cur_le,
                'alpha': cur_alpha,
                'n_clients': n_clients,
                'table': table,
            })
            i = j
            continue
        i += 1
    return blocks


if __name__ == '__main__':
    out = {}
    for path in sys.argv[1:]:
        out[path] = parse(path)
    print(json.dumps(out, indent=1))
