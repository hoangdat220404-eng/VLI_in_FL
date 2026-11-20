
import argparse

def args_parser():
    
    # Optimization options

    parser = argparse.ArgumentParser(description='PyTorch Attacking Labels')
    parser.add_argument('--epochs', default= 20, type=int, metavar='N',
                        help='number of total epochs to run')
    
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='training batchsize')

    parser.add_argument('-d', '--dataset', default='CIFAR10', type=str, help='SVHN, CIFAR10, CIFAR100, Tiny-ImageNet')

    parser.add_argument('-p', '--prop', default=0.1, type=float, help='proportion of aux data')

    parser.add_argument('-m', '--model', default='resnet18', type=str, help='vgg, resnet18, resnet54, lenet')

    parser.add_argument('--activation', default='relu', type=str, help='relu, tanh, elu, selu, silu')

    parser.add_argument('--hidden', default=512, type=int, help='dimension of the hidden embeddings')

    parser.add_argument('-n', '--n_clients', default=10, type=int)
    parser.add_argument('--vli_star_gmax', default='max', choices=('max', 'mean+', 'med+', 'gap', 'otsu'),
                        help="VLI*: mau so G*. 'max' = ban goc; 'mean+'/'med+' = trung binh/trung vi cua cac G_j > 0 (ben voi nham lan khong deu); "
                             "'gap'/'otsu' = trung binh cum G_j > 0 tren cung, tach bang khoang trong lon nhat / Otsu")
    parser.add_argument('--vli_star_diag', action='store_true',
                        help='VLI*: in chan tren oracle cua G_max va cac bien the phan vi (dung nhan that, chi de phan tich)')
    
    parser.add_argument('--sampling_rate', default=1.0, type=float, help='sampling rate for clients')
    
    parser.add_argument('--n_classes', default = 10, type=int)

    parser.add_argument('--alpha', default=0.5, type=float)
    
    parser.add_argument('--local_epochs', default = 1, type=int, metavar='N')
    
    
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')

    parser.add_argument('--scheme', default='RLU',type=str,
                        help='RLU, iRLG, LLGp, ZLGp, VLI, VLI_star, All')

    parser.add_argument('--fl_scheme', default='fedavg', type=str,
                        help='fedavg, fedprox, feddyn, scaffold')

    parser.add_argument('--mu', default=0.0, type=float,
                        help='hyper-parameter for fedprox, feddyn, scaffold')

    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    parser.add_argument('--momentum', default=0.0, type=float, metavar='M',
                    help='momentum')
    
    parser.add_argument('--weight_decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

    
    parser.add_argument('--alpha_coefficient', '--ac', default=1e-2, type=float,
                    metavar='W', help='alpha coefficient (only for FedDyn)')
    
    parser.add_argument('-c', '--checkpoint', default='./checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint')

    parser.add_argument('-cpph', '--check_point_path', default='resnet18_CIFAR10_acc_92.63.pth', type=str, metavar='PATH',
                    help='global-model checkpoint to attack; pass an empty string to attack an untrained model (official default)')

    parser.add_argument('--rlu_posterior_fix', action='store_true',
                        help='RLU posterior search: compare against the full new_mu vector instead of the official '
                             'new_mu[last_epoch] scalar (a bug in the official code). Off by default = official behaviour.')

    parser.add_argument('--rlu_diag', action='store_true',
                        help='RLU: in ||O||^2, kappa, A/n va do dich cua tam vong hai so voi softmax(new_mu). '
                             'Chi doc tin hieu + aux, khong dung nhan that.')

    # ---- co cho cac thuc nghiem kiem chung chan bay (them moi, khong doi mac dinh) ----
    parser.add_argument('--load_ckpt', action='store_true',
                        help='Nap --check_point_path vao global model va BO QUA vong tien huan luyen. '
                             'Khong bat thi giu nguyen hanh vi cu (huan luyen tu dau --pretrain_rounds vong).')
    parser.add_argument('--pretrain_rounds', type=int, default=None,
                        help='So vong FedAvg tien huan luyen khi KHONG dung --load_ckpt. Mac dinh None = giu nguyen gia tri trong main.py.')
    parser.add_argument('--target_acc', type=float, default=None,
                        help='Dung tien huan luyen ngay khi training accuracy top-1 (%%) vuot nguong nay; '
                             '--pretrain_rounds khi do la so vong toi da.')
    parser.add_argument('--oracle_stats', action='store_true',
                        help='ORACLE (dung du lieu that cua client thay cho aux trong estimate_static_*). '
                             'Khong phai tan cong kha thi - chi de tach "uoc luong toi" khoi "hong cau truc".')
    parser.add_argument('--collapse_diag', action='store_true',
                        help='Do khoang cach tu dau ra toi diem tham chieu khi tin hieu = 0, va irec cua chinh diem do.')
    parser.add_argument('--inspect', action='store_true',
                        help='Loat kiem tra cho RLU/ZLGp/LLGp moi cua so: (1) neo do tren mo hinh GLOBAL '
                             'thay vi mo hinh da troi -> tach kenh "trong so ghi nho" khoi kenh gradient; '
                             '(2) thay gradient bang hoan vi toa do / gradient client khac / nhan ti le '
                             '-> do xem gradient co that su duoc doc khong.')

    args = parser.parse_args()
    return args

