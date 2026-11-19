#!/usr/bin/env bash
# Thuc nghiem kiem chung chan bay. Chay tu thu muc VLI_in_FL.
#   bash scripts/exps_collapse.sh E0|E1|E2|E3|E4
set -u
OUT=logs_collapse; mkdir -p "$OUT"
R18="--dataset CIFAR10 --model resnet18 --hidden 512 --n_classes 10"
C100="--dataset CIFAR100 --model resnet18 --hidden 512 --n_classes 100"
VGG="--dataset CIFAR10 --model vgg16 --hidden 512 --n_classes 10"
COMMON="--local_epoch 10 --momentum 0.0 --alpha 0.5 --batch_size 64"

case "${1:-}" in

# --- E0: kiem checkpoint co nap duoc that khong ------------------------------
E0)
  python main.py --scheme All --load_ckpt \
    --check_point_path resnet18_CIFAR10_acc_87.36_29.pth \
    $R18 $COMMON 2>&1 | tee "$OUT/E0_sanity.txt"
  echo "--- phai thay dong '==> loaded global model checkpoint' va 'so vong tien huan luyen: 0' ---"
  ;;

# --- E1+E2: quet theo do chinh xac, kem khoang cach toi diem sup do ----------
E1)
  for f in resnet18_CIFAR10_acc_*.pth; do
    acc=$(echo "$f" | sed 's/.*acc_\([0-9.]*\)_.*/\1/')
    echo "=== $f (acc=$acc) ==="
    python main.py --scheme All --load_ckpt --check_point_path "$f" \
      --collapse_diag --rlu_diag $R18 $COMMON 2>&1 | tee "$OUT/E1_r18_c10_acc${acc}.txt"
  done
  ;;

# --- E3: phep thu nhan qua bang gamma (chi doi hang so ben tan cong) ---------
E3)
  CK=resnet18_CIFAR10_acc_87.36_29.pth
  for g in 0.01 0.3 0.5 0.7 0.9; do
    echo "=== gamma=$g ==="
    python main.py --scheme All --load_ckpt --check_point_path "$CK" \
      --gamma "$g" --collapse_diag $R18 $COMMON 2>&1 | tee "$OUT/E3_gamma${g}.txt"
  done
  ;;

# --- E4: ablation oracle (thay aux bang du lieu that cua client) ------------
E4)
  for CK in resnet18_CIFAR10_acc_27.74_0.pth resnet18_CIFAR10_acc_87.36_29.pth; do
    acc=$(echo "$CK" | sed 's/.*acc_\([0-9.]*\)_.*/\1/')
    for mode in "" "--oracle_stats"; do
      tag=$([ -z "$mode" ] && echo aux || echo oracle)
      echo "=== acc=$acc  nguon thong ke=$tag ==="
      python main.py --scheme All --load_ckpt --check_point_path "$CK" \
        $mode --collapse_diag $R18 $COMMON 2>&1 | tee "$OUT/E4_acc${acc}_${tag}.txt"
    done
  done
  ;;

# --- E5: ResNet18 vs VGG16 khop theo DO CHINH XAC, khong theo epoch ---------
E5)
  python main.py --scheme All --load_ckpt --check_point_path resnet18_CIFAR10_acc_80.27_16.pth \
    --rlu_diag --collapse_diag $R18 $COMMON 2>&1 | tee "$OUT/E5_r18_80.27.txt"
  python main.py --scheme All --load_ckpt --check_point_path vgg16_CIFAR10_acc_80.42_11.pth \
    --rlu_diag --collapse_diag $VGG $COMMON 2>&1 | tee "$OUT/E5_vgg_80.42.txt"
  python main.py --scheme All --load_ckpt --check_point_path resnet18_CIFAR10_acc_87.00_28.pth \
    --rlu_diag --collapse_diag $R18 $COMMON 2>&1 | tee "$OUT/E5_r18_87.00.txt"
  python main.py --scheme All --load_ckpt --check_point_path vgg16_CIFAR10_acc_86.75_16.pth \
    --rlu_diag --collapse_diag $VGG $COMMON 2>&1 | tee "$OUT/E5_vgg_86.75.txt"
  ;;

# --- E6: phu thuoc vao C (CIFAR100, bay cua ZLG+ phai chat hon ~10 lan) -----
E6)
  for f in resnet18_CIFAR100_acc_*.pth; do
    acc=$(echo "$f" | sed 's/.*acc_\([0-9.]*\)_.*/\1/')
    python main.py --scheme All --load_ckpt --check_point_path "$f" \
      --collapse_diag --alpha 0.1 $C100 --local_epoch 10 --momentum 0.0 --batch_size 64 \
      2>&1 | tee "$OUT/E6_c100_acc${acc}.txt"
  done
  ;;

*) echo "dung: bash scripts/exps_collapse.sh E0|E1|E3|E4|E5|E6"; exit 1;;
esac
