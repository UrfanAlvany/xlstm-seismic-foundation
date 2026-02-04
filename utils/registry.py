import models.phasenet_wrapper

optimizer = {
    "adam":    "torch.optim.Adam",
    "adamw":   "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd":     "torch.optim.SGD",
    "lamb":    "src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant":        "transformers.get_constant_schedule",
    "plateau":         "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step":            "torch.optim.lr_scheduler.StepLR",
    "multistep":       "torch.optim.lr_scheduler.MultiStepLR",
    "cosine":          "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup":   "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup":   "transformers.get_cosine_schedule_with_warmup",
    'double_linear':   "utils.custom_schedulers.DoubleLinearScheduler"
}

model = {
    "lstm": "models.lstm_baseline.LSTMSequenceModel",
    "identity": "models.simple_test_models.IdentityModel",
    "phase-net": "models.phasenet_wrapper.PhaseNetWrapper",
    "contrastive": "models.contrastive_wrapper.ContrastiveModel",
    "xlstm": "models.xLSTM.OfficialXLSTMModel",
    # xLSTM U-Net backbone (no classification head). Uses TFLA kernels when available.
    "xlstm-unet": "models.xlstm_unet.xLSTMUNetBackbone",
    # Alias for clarity if referenced elsewhere
    "xlstm-backbone": "models.xlstm_unet.xLSTMUNetBackbone",
}

dataset = {
    "mnist": "dataloaders.MNISTdataloader.MNISTdataset",
    "sine": "dataloaders.simple_waveform.SineWaveLightningDataset",
    "costarica-small": "dataloaders.costa_rica_small.CostaRicaSmallLighting",
    "costarica-long-seq": "dataloaders.costa_rica_quantized.CostaRicaQuantizedLightning",
    "costarica-bpe": "dataloaders.costa_rica_bpe.CostaRicaBPELightning",
    "costarica-enc-dec": "dataloaders.costa_rica_quantized.CostaRicaEncDecLightning",
    "ethz-auto-reg": "dataloaders.seisbench_auto_reg.SeisBenchAutoReg",
    "ethz-phase-pick": "dataloaders.seisbench_auto_reg.SeisBenchPhasePick",
    "audio-dataset": "dataloaders.audio_loader.AudioDatasetLit",
    "foreshock-aftershock": "dataloaders.foreshock_aftershock_lit.ForeshockAftershockLitDataset",
    "insight": "dataloaders.insight_loader.InsightDataLit",
    "insight-2": "dataloaders.insight_loader_2.InsightDataLit2",
    "insight-pretrain": "dataloaders.insight_loader.InsightPretrainDataLit",
}

preloadable_datasets = [
    "ethz-auto-reg",
    "ethz-phase-pick",
]
