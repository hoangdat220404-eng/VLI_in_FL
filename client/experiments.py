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
from client.attacks import (rlu_rho_mean, ZLGp_attack, LLGp_attack, LLG_attack,
                            RLU_attack)


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


# ===========================================================================
#  INSPECTION BATTERY (them moi, khong sua code cu)
#
#  Hai phep do tra loi hai cau hoi khac nhau:
#
#  1. `anchor_provenance` -- NEO DEN TU DAU?
#     RLU/ZLGp/LLGp dung tham so `model` DUY NHAT mot cho: de goi
#     estimate_static_*. Nen truyen vao mot ban copy da nap global_weights
#     thi ta doi DUNG cai neo, giu nguyen gradient that. Neu irec sut ve muc
#     deu => thong tin cua neo den tu viec TRONG SO GHI NHO phan phoi cua
#     client trong m buoc local, khong phai tu gradient.
#
#  2. `grad_ablation` -- GRADIENT CO THAT SU DUOC DOC KHONG?
#     Thay G bang: hoan vi toa do (giu nguyen moi do lon, pha huy canh nhan),
#     G cua client khac, hoac G nhan ti le. Neu irec khong doi => gradient
#     chi la vat trang tri. Manh hon phep dat G=0 vi zero co the roi vao
#     nhanh code suy bien.
#
#  CA HAI deu dung nhan that de tinh irec, giong moi phep do metric khac.
# ===========================================================================

_FOREIGN = {}   # kho gradient "ngoai lai" de lam ablation cross-client


def stash_foreign(w_grad, b_grad):
    """Luu gradient cua cua so DAU TIEN gap duoc, dung lam gradient ngoai lai."""
    if 'w' not in _FOREIGN:
        _FOREIGN['w'] = w_grad.detach().clone()
        _FOREIGN['b'] = b_grad.detach().clone()


def _seeded(fn):
    """Chay fn() voi cung mot trang thai RNG, roi phuc hoi.

    Bat buoc: estimate_static_* lay mau aux ngau nhien (np.random.choice +
    DataLoader shuffle). Khong ghim RNG thi hai bien the se dung hai tap aux
    khac nhau va phep so sanh mat sach.
    """
    st_t, st_n = torch.get_rng_state(), np.random.get_state()
    try:
        return fn()
    finally:
        torch.set_rng_state(st_t)
        np.random.set_state(st_n)


def _run_three(model, w_grad, b_grad, latent_dim, mu, aux_dataset, args,
               O_bar, pj, impact, offset):
    """Chay RLU/ZLGp/LLGp tren mot `model` cho truoc. Tra ve dict counts."""
    G1 = torch.sum(w_grad, dim=-1)
    out = {}
    try:
        out['RLU'] = _seeded(lambda: RLU_attack(
            model=copy.deepcopy(model), w_grad_epochs=w_grad,
            b_grad_epochs=b_grad, latent_dim=latent_dim, mu=mu,
            aux_dataset=aux_dataset, args=args))
    except Exception as exc:
        print('  [inspect] RLU hong: %s' % exc)
    try:
        out['ZLGp'] = _seeded(lambda: ZLGp_attack(
            model=copy.deepcopy(model), gradients_for_prediction=G1.clone(),
            O_bar=O_bar, pj=pj, aux_dataset=aux_dataset, args=args))
    except Exception as exc:
        print('  [inspect] ZLGp hong: %s' % exc)
    try:
        out['LLGp'] = _seeded(lambda: LLGp_attack(
            model=copy.deepcopy(model), gradients_for_prediction=G1.clone(),
            impact=impact, offset=offset, aux_dataset=aux_dataset, args=args))
    except Exception as exc:
        print('  [inspect] LLGp hong: %s' % exc)
    return out


def run_inspection(args, model_drift, model_global, w_grad, b_grad, latent_dim,
                   mu, aux_dataset, num_instances, n,
                   O_bar=None, pj=None, impact=None, offset=None, seed=0):
    """Chay toan bo loat kiem tra cho mot cua so. Tra ve {ten_bien_the: {pp: irec}}."""
    common = dict(latent_dim=latent_dim, mu=mu, aux_dataset=aux_dataset,
                  args=args, O_bar=O_bar, pj=pj, impact=impact, offset=offset)
    res = {}

    # --- goc: neo tren mo hinh DA TROI, gradient that -------------------
    res['base'] = _run_three(model_drift, w_grad, b_grad, **common)

    # --- PHEP 1: neo tren mo hinh GLOBAL, gradient that ----------------
    res['anchor_global'] = _run_three(model_global, w_grad, b_grad, **common)

    # --- PHEP 2a: hoan vi toa do lop (giu do lon, pha canh nhan) -------
    #     hoan vi DONG BO ca w_grad (theo hang) va b_grad de O = mean(dW/db)
    #     khong bi pha ve cau truc, chi mat canh voi nhan that.
    rng = np.random.RandomState(seed)
    perm = torch.as_tensor(rng.permutation(args.n_classes), dtype=torch.long)
    res['shuffle'] = _run_three(model_drift, w_grad[perm].clone(),
                                b_grad[perm].clone(), **common)

    # --- PHEP 2b: gradient cua client khac -----------------------------
    if 'w' in _FOREIGN and _FOREIGN['w'].shape == w_grad.shape:
        res['foreign'] = _run_three(model_drift, _FOREIGN['w'].clone(),
                                    _FOREIGN['b'].clone(), **common)

    # --- PHEP 2c: nhan ti le (bat bien thang do?) ----------------------
    for s in (0.1, 10.0):
        res['scale%g' % s] = _run_three(model_drift, (w_grad * s).clone(),
                                        (b_grad * s).clone(), **common)

    return {k: {m: irec_of(c, num_instances, n) for m, c in v.items()}
            for k, v in res.items()}


def print_inspection(table):
    """In mot dong moi bien the, de parse lai bang grep '[inspect]'."""
    order = ['base', 'anchor_global', 'shuffle', 'foreign', 'scale0.1', 'scale10']
    for name in order:
        if name not in table:
            continue
        row = table[name]
        parts = ['%s=%.4f' % (m, row[m]) for m in ('RLU', 'ZLGp', 'LLGp') if m in row]
        print('  [inspect] %-14s %s' % (name, ' '.join(parts)))
