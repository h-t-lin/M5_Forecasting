# Reading/Writing Data
import os
import argparse
import math

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# For dataset
import random
from dataset import M5_Dataset

# Models
from SeqModel.transformer import ForcastingTransformer

# For training
from Utils.loss_fn import loss_settings
from Utils.optim import optimization_settings
from Utils.utils import train, inference

# Other needs:
from configuration import Config
from Utils.plot_fn import plot_loss_curve
import Utils.Assistant_func.settings as settings
from Utils.Assistant_func.initialization import Path_init, Display_info

def parse_args():
    parser = argparse.ArgumentParser(description='Training Process of Low-dose Tau PET Enhancement Project')
    parser.add_argument(
        '--desc', default='/', dest='Description', help='description', type=str)
    parser.add_argument(
        '--load', default=-1, dest='loadID', help='ID of the result you want to load, use negative number to load nothing.', type=int)
    parser.add_argument(
        '--loadopt', default=0, dest='loadOPT', help='load params for optimizer or not (0/1)', type=int)
    parser.add_argument(
        '--fold', default=None, dest='CVfold', help='cross-validation fold', type=int)
    parser.add_argument(
        '--lr', default=None, dest='learning_rate', help='cross-validation fold', type=float)
    parser.add_argument(
        '--input_days', default=None, dest='input_days', help='cross-validation fold', type=int)
    parser.add_argument(
        '--scheduler', default=None, dest='scheduler', help='cross-validation fold', type=str)
    parser.add_argument(
        "--null", default='------------------------', dest="NULL")
    args, rest = parser.parse_known_args()
    return args, rest

def main():

    # Initialization----
        # create Config
        # initialize path
        # set device and seed for reproducibility
    args, restargs = parse_args()
    config = Config(args, restargs)
    config.import_paras("hyperparameters.yaml")
    path = Path_init("path_file.yaml", config.trainID, config.loadID, create_new=True)
    # config.import_paras(os.path.join(path['data_path'], path['dataset_cfg']))
    if config.loadID>=0:
        config.import_training_paras(os.path.join(path['load_config_path'], path['training_cfg']))
    config.save(os.path.join(path['result_path'], path['training_cfg']))
    Display_info(vars(config), path, create_new=True)
    settings.set_seed(config.seed)
    config.device = settings.set_device(config.device)

    # Dataloader----
        # dataset splitting for training/validation/test
        # create dataset, dataloader
    data_path = path['data_path']

    train_set = M5_Dataset(data_path, Twin=[config.input_days,config.input_days], Tpred=config.output_days, is_inference=False)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.CPU_workers)

    valid_set = M5_Dataset(data_path, Twin=[config.input_days,config.input_days], Tpred=config.output_days*config.regressive_rounds, is_inference=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=config.CPU_workers)

    # display_set = LDTAU_FE_dataset(data_path, random.sample(dataset_filedict['valid'], config.imgrid_num), config, istest=False, random_pick=True)
    # display_loader = DataLoader(display_set, batch_size=config.batch_size, shuffle=False, num_workers=config.CPU_workers)

    # Setup----
        # define model, criterion, optimizer, lr scheduler
    print("\n*\n*\n----START TRAINING----\n")
    model = ForcastingTransformer(input_dim=3049, store_dim=10, date_dim=11, model_dim=config.transformer_dim, nhead=config.n_heads,\
                                  num_encoder_layers=config.enc_layers, num_decoder_layers=config.dec_layers, dim_feedforward=2048).to(config.device)

    criterion = loss_settings(loss_type='MSE')
    optimizer, lr_scheduler = optimization_settings(model.parameters(), config)
    if config.loadID>=0:
        checkpoint = torch.load(path['load_model_path'], map_location=config.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Training----
    stopper = 0
    loss_record = {"train": [], "valid": [], "best": math.inf}
    eta = torch.sigmoid(torch.tensor([7.5*(ep/config.n_epochs-0.5) for ep in range(config.n_epochs)], dtype=torch.float32))
    for epoch in range(1, config.n_epochs+1):
        print(f"Epoch:[{epoch}/{config.n_epochs}]")

        # Training phase
        # loss = train(train_loader, model, criterion, optimizer, lr_scheduler, config.device, eta_L=eta[epoch-1].item())
        # loss_record["train"].append( loss / len(train_loader.dataset) )

        # Validation phase
        loss = inference(valid_loader, model, criterion, config.device, num_round=4, display_text="  Validation")
        loss_record["valid"].append( loss / len(valid_loader.dataset) )

        print("  Loss:    training loss: %-9.3e    validation loss: %-9.3e" % (loss_record['train'][-1], loss_record['valid'][-1]))

        # save best model (with lowest loss)...
        if loss_record["valid"][-1] < loss_record["best"]:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch
                }, path["model_path"])
            loss_record["best"] = loss_record["valid"][-1]
            stopper = 0
            print("(Model saved.)")
        else:
            stopper += 1
            print(f"(Best model was saved at epoch [{epoch-stopper}].)")
            if stopper > config.early_stop:
                print("\nTraining is early stopped since there has been no progression of loss for a long while.")
                break
            elif stopper > config.early_stop - 5:
                print(f"(Early stop left: {config.early_stop-stopper+1} epoch(s).)")
        plot_loss_curve(loss_record, epoch, path["result_path"])

    print(f"*\n*\n----FINISHED----\n")


if __name__ == "__main__":
    main()