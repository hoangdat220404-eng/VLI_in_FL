# -*- coding: utf-8 -*-
"""Cong cu cho cac thuc nghiem kiem chung chan bay (them moi, khong sua code cu).

Hai thu o day:

1. `collapse_reference`  -- tinh diem tham chieu Q(R(c)): dau ra cua CUNG duong ong
   do khi tin hieu cua client bi dat bang 0. Chan bay noi dau ra that phai nam gan
   diem nay, va khoang cach co lai khi mo hinh duoc huan luyen tot.

2. `oracle_dataset`      -- tra ve tap du lieu that cua client de thay cho aux trong
   cac ham estimate_static_*. DAY LA ORACLE, khong phai mot tan cong kha thi.

Ca hai chi phuc vu phan tich. `irec_of` co dung nhan that, giong moi phep do metric.
"""

import numpy as np
import torch
import copy
import scipy.special

from client.client_utils import (
    estimate_static_RLU,
    estimated_entropy_from_grad,
    estimate_static_LLG,
    estimate_static_ZLG,
)
from client.attacks import rlu_rho_mean, ZLGp_attack, LLGp_attack, LLG_attack


def oracle_dataset(client):
    """Tap du lieu that cua client (thay cho aux). Dung cho ablation oracle."""
    return client.trainloader.dataset


def irec_of(counts, num_instances, n):
    """irec = sum_j min(N_hat_j, N_j) / n -- cung dinh nghia voi accuracy_calculation."""
    c = np.asarray(counts, dtype=float)
    N = np.asarray(num_instances, dtype=float)
    return float(np.minimum(c, N).sum() / n)


def _l1(a, b, n):
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum() / n)


def collapse_reference(args, model, aux_dataset, n_classes, n, num_instances,
                       outputs=None, O_bar=None, pj=None, impact=None, offset=None):
    """Tinh Q(R(c)) cho tung phuong phap bang cach dat tin hieu = 0.

    `outputs` la dict {ten_phuong_phap: counts} cua lan chay THAT, de tinh khoang cach.
    Tra ve dict {ten: (d, irec_tham_chieu)} voi d = ||N_hat - Q(R(c))||_1 / n.
    """
    zero_vec = torch.zeros(n_classes)
    ref = {}

    # ---- RLU: dang chuan vong mot voi tin hieu = 0 ----
    try:
        new_mu, _ = estimate_static_RLU(args, copy.deepcopy(model), aux_dataset)
        shift = scipy.special.softmax(new_mu) * rlu_rho_mean(args)
        ref['RLU'] = estimated_entropy_from_grad(args, shift, [0.0] * n_classes, n)
    except Exception as exc:
        print('  [collapse] RLU: khong tinh duoc tham chieu (%s)' % exc)

    # ---- ZLG+ ----
    try:
        ref['ZLGp'] = ZLGp_attack(model=copy.deepcopy(model), gradients_for_prediction=zero_vec.clone(),
                                  O_bar=O_bar, pj=pj, aux_dataset=aux_dataset, args=args)
    except Exception as exc:
        print('  [collapse] ZLGp: khong tinh duoc tham chieu (%s)' % exc)

    # ---- LLG+ / LLG ----
    try:
        ref['LLGp'] = LLGp_attack(model=copy.deepcopy(model), gradients_for_prediction=zero_vec.clone(),
                                  impact=impact, offset=offset, aux_dataset=aux_dataset, args=args)
    except Exception as exc:
        print('  [collapse] LLGp: khong tinh duoc tham chieu (%s)' % exc)
    try:
        ref['LLG'] = LLG_attack(model=copy.deepcopy(model),
                                gradients_for_prediction=zero_vec.clone(), args=args)
    except Exception as exc:
        print('  [collapse] LLG: khong tinh duoc tham chieu (%s)' % exc)

    out = {}
    for name, r in ref.items():
        d = _l1(outputs[name], r, n) if (outputs and name in outputs) else float('nan')
        out[name] = (d, irec_of(r, num_instances, n))
    return out


def print_collapse(table):
    """In mot dong cho moi phuong phap: khoang cach toi diem tham chieu + irec cua diem do."""
    if not table:
        return
    parts = []
    for name in sorted(table):
        d, ir = table[name]
        parts.append('%s: d=%.4f irec_ref=%.4f' % (name, d, ir))
    print('  [collapse] ' + ' | '.join(parts))
