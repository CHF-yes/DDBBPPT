"""RF-DETR-M fixed 864x1536 RGB fine-tuning on CHF split s42."""
import json
from pathlib import Path
import torch
from pytorch_lightning import Callback
from rfdetr import RFDETRMedium
from rfdetr.datasets import coco
from rfdetr.datasets._torchvision import Compose, Resize, RandomHorizontalFlip
from rfdetr.datasets.transforms import Normalize
from torchvision.transforms.v2 import ToImage, ToDtype

ROOT = Path('/root/autodl-tmp/chf_arch_baselines_20261003')
OUT = ROOT / 'runs/rfdetr_m_highres_864x1536_20261003'
DATA = Path('/root/autodl-tmp/data/chf_arch_rgb_coco_s42')
WEIGHT = ROOT / 'runs/rfdetr_m/checkpoint_best_ema.pth'
SHAPE = (864,1536)

def rectangular_transforms(image_set, resolution, **kwargs):
    ops = [Resize(SHAPE)]
    if image_set == 'train':
        ops.append(RandomHorizontalFlip(p=0.5))
    ops.extend([ToImage(), ToDtype(torch.float32, scale=True), Normalize()])
    return Compose(ops)

class VerifyCanvas(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 0:
            shape = tuple(batch[0].tensors.shape[-2:])
            assert shape == SHAPE, ('train', shape)
            print('CHF actual training canvas:', shape, flush=True)
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            shape = tuple(batch[0].tensors.shape[-2:])
            assert shape == SHAPE, ('validation', shape)
            print('CHF actual validation canvas:', shape, flush=True)

def main():
    manifest=json.loads((DATA/'export_manifest.json').read_text())
    assert manifest['split_sha256']=='4165ce54e9b34be51ec3f87d3065977eee310da3ce700fa4b5b19a85202ba00a'
    assert WEIGHT.is_file()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True,exist_ok=True)
    record={'canvas':SHAPE,'epochs':12,'batch_size':1,'grad_accum_steps':8,'eval_batch_size':1,'lr':2e-5,'lr_encoder':2e-6,'initial_weight':str(WEIGHT),'split_sha256':manifest['split_sha256'],'modalities':'RGB','augmentation':'horizontal flip p=0.5; fixed rectangular resize; no multi-scale/crop','resume':'weight initialization; new optimizer and EMA'}
    (OUT/'chf_run_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    coco.make_coco_transforms_square_div_64=rectangular_transforms
    coco.make_coco_transforms=rectangular_transforms
    import rfdetr.training as training
    original_build_trainer = training.build_trainer
    def build_with_canvas_check(*args, **kwargs):
        trainer = original_build_trainer(*args, **kwargs)
        trainer.callbacks.append(VerifyCanvas())
        return trainer
    training.build_trainer = build_with_canvas_check
    torch.set_num_threads(8)
    model=RFDETRMedium(pretrain_weights=str(WEIGHT),gradient_checkpointing=True)
    model.train(dataset_dir=str(DATA),output_dir=str(OUT),class_names=manifest['classes'],epochs=12,resolution=1536,batch_size=1,grad_accum_steps=8,eval_batch_size=1,lr=2e-5,lr_encoder=2e-6,amp_dtype='bf16',seed=42,num_workers=8,multi_scale=False,scale_jitter=False,checkpoint_interval=4,skip_best_epochs=0,early_stopping=False)

if __name__=='__main__':
    main()
