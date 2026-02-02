import argparse
import evaluation.pick_eval as pe
from models.evaluation_wrapper import PhasePickerLit
import models.benchmark_models as bm


def get_pretrained_model(model_name, training_dataset):
    if model_name == 'PhaseNet':
        model = bm.PhaseNetLit(pretrained=True, pretrain_data=training_dataset)
    elif model_name == 'EQTransformer':
        model = bm.EQTransformerLit(pretrained=True, pretrain_data=training_dataset)
    elif model_name == 'DPPPicker':
        model = bm.DPPPickerLit(pretrained=True, pretrain_data=training_dataset)
    elif model_name == 'DPPDetector':
        model = bm.DPPDetectorLit(pretrained=True, pretrain_data=training_dataset)
    elif model_name == 'GPD':
        model = bm.GPDLit(pretrained=True, pretrain_data=training_dataset)
    elif model_name == 'BasicPhase':
        model = bm.BasicPhaseAELit(pretrained=True, pretrain_data=training_dataset)
    else:
        print(f'Unknown model: {model_name}')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate evaluation targets. See the docstring for details."
    )
    parser.add_argument(
        "model_name", type=str, default='PhaseNet', help="Name of pretrained model"
    )
    parser.add_argument(
        'train_dataset', type=str, default='ethz', help="Name of the dataset used to pretrain the model"
    )
    parser.add_argument(
        '--target_dataset', type=str, default='ETHZ', help="Name of the dataset to evaluate"
    )
    parser.add_argument(
        '--num_workers', type=int, default=0, help="Number of workers for dataloader"
    )
    parser.add_argument(
        '--sets', type=str, default='train,dev,test', help="Name of the target splits"
    )
    args = parser.parse_args()

    target_path = 'evaluation/eval_tasks/' + args.target_dataset

    model_name = args.model_name
    model_name = model_name.replace(' ', '')
    if model_name == 'all':
        model_names = ['PhaseNet', 'EQTransformer', 'BasicPhase', 'GPD'] #'DPPPicker', 'DPPDetector',
    else:
        model_names = model_name.split(',')
    training_dataset = args.train_dataset

    #model = PhasePickerLit(ckpt_path=None, pretrained_name=model_name, pretrained_dataset=training_dataset)

    for m in model_names:
        print('*' * 32)
        print(f'\nEvaluating model: {m}')
        print('*' * 32, '\n')

        model = get_pretrained_model(m, training_dataset=training_dataset)

        pe.save_pick_predictions(
            model=model,
            target_path=target_path,
            ckpt_path=f'pretrained_benchmark/{m}_{args.train_dataset}',
            sets=args.sets,
            save_tag='eval',
            batch_size=64,
            num_workers=args.num_workers
        )


