import argparse
import evaluation.pick_eval as pe
from models.evaluation_wrapper import PhasePickerLit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate evaluation targets. See the docstring for details."
    )
    parser.add_argument(
        "checkpoint", type=str, help="Path to model checkpoint"
    )
    parser.add_argument(
        '--target_dataset', type=str, default='ETHZ', help="Name of the target dataset e.g. ETHZ"
    )
    parser.add_argument(
        '--num_workers', type=int, default=0, help="Number of workers for dataloader"
    )
    parser.add_argument(
        '--sets', type=str, default='train,dev,test', help="Name of the target splits"
    )
    parser.add_argument(
        '--norm_type', type=str, default='peak', help="Type of amplitude norm for data loader"
    )
    parser.add_argument(
        '--save_tag', type=str, default='eval', help="Subfolder tag under evals/ (e.g., eval, eval_last)"
    )
    args = parser.parse_args()

    sets = 'train,dev,test'
    ckpt_path = args.checkpoint

    model = PhasePickerLit(ckpt_path, norm_type=args.norm_type)

    if '.ckpt' in ckpt_path:
        ckpt_path = '/'.join(ckpt_path.split('/')[:-2])
    target_path = 'evaluation/eval_tasks/' + args.target_dataset

    pe.save_pick_predictions(
        model=model,
        target_path=target_path,
        ckpt_path=ckpt_path,
        sets=args.sets,
        save_tag=args.save_tag,
        batch_size=64,
        num_workers=args.num_workers
    )
