import wandb
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import pyqg_explorer.systems.regression_systems as reg_sys
import pyqg_explorer.models.fcnn as fcnn
import pyqg_explorer.util.performance as performance
import pyqg_explorer.dataset.forcing_dataset as forcing_dataset

config=reg_sys.config
config["epochs"]=100

emulator_dataset=forcing_dataset.OfflineDataset("/scratch/cp3759/pyqg_data/sims/torchqg_sims/0_step/all.nc",seed=config["seed"],subsample=config["subsample"],drop_spin_up=config["drop_spin_up"])

## Need to save renormalisation factors for when the CNN is plugged into pyqg
config["q_mean_upper"]=emulator_dataset.q_mean_upper
config["q_mean_lower"]=emulator_dataset.q_mean_lower
config["q_std_upper"]=emulator_dataset.q_std_upper
config["q_std_lower"]=emulator_dataset.q_std_lower
config["s_mean_upper"]=emulator_dataset.s_mean_upper
config["s_mean_lower"]=emulator_dataset.s_mean_lower
config["s_std_upper"]=emulator_dataset.s_std_upper
config["s_std_lower"]=emulator_dataset.s_std_lower

train_loader = DataLoader(
    emulator_dataset,
    num_workers=10,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(emulator_dataset.train_idx),
)
valid_loader = DataLoader(
    emulator_dataset,
    num_workers=10,
    batch_size=config["batch_size"],
    sampler=SubsetRandomSampler(emulator_dataset.valid_idx),
)

## Define CNN module
model=fcnn.FCNN(config)

## Loss function defined in a RegressionSystem module
system=reg_sys.RegressionSystem(model,config)

wandb.init(project="torch_offline", entity="m2lines",config=config)
## Add number of parameters of model to config
wandb.config["theta learnable parameters"]=sum(p.numel() for p in model.parameters())
wandb.watch(model, log_freq=1)

logger = WandbLogger()
## This will log learning rate to wandb
lr_monitor=LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(
    accelerator="auto", ## Use GPU if lightning can find one
    max_epochs=config["epochs"],
    logger=logger,
    enable_progress_bar=False,
    callbacks=[lr_monitor]
    )

trainer.fit(system, train_loader, valid_loader)

## Run performance tests, and upload figures to wandb
perf=performance.ParameterizationPerformance(model,valid_loader,threshold=5000)

dist_fig=perf.get_distribution_2d()
figure_dist=wandb.Image(dist_fig)
wandb.log({"Distributions": figure_dist})

power_fig=perf.get_power_spectrum()
figure_power=wandb.Image(power_fig)
wandb.log({"Power spectra": figure_power})

field_fig=perf.get_fields()
figure_field=wandb.Image(field_fig)
wandb.log({"Random fields": figure_field})

subgrid_fig=perf.subgrid_energy()
figure_subgrid=wandb.Image(subgrid_fig)
wandb.log({"Subgrid energy": figure_subgrid})

wandb.finish()