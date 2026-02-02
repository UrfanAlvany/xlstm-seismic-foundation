import argparse
import torch


def load_state(path):
    ckpt = torch.load(path, map_location='cpu')
    if 'state_dict' not in ckpt:
        raise ValueError(f'No state_dict in {path}')
    return ckpt


def average_ckpts(paths):
    base = load_state(paths[0])
    sd = base['state_dict']
    keys = list(sd.keys())
    agg = {k: sd[k].clone().float() for k in keys if torch.is_tensor(sd[k])}
    n = 1
    for p in paths[1:]:
        ck = load_state(p)['state_dict']
        for k in agg:
            agg[k] += ck[k].float()
        n += 1
    for k in agg:
        agg[k] /= n
    base['state_dict'] = {k: (agg[k].to(sd[k].dtype) if k in agg else sd[k]) for k in sd}
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output ckpt path')
    ap.add_argument('ckpts', nargs='+', help='input ckpt paths (2 or more)')
    args = ap.parse_args()
    if len(args.ckpts) < 2:
        raise SystemExit('Need at least 2 checkpoints to average')
    avg = average_ckpts(args.ckpts)
    torch.save(avg, args.out)
    print(f'Saved averaged checkpoint: {args.out}')


if __name__ == '__main__':
    main()

