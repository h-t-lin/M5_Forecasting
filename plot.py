# Numerical Operations
import numpy as np

# Reading/Writing Data
import os

# Pytorch
import torch
from torch.utils.data import DataLoader

# import config, path
from configuration import Config
import argparse

# import dataset
from dataset import M5_Dataset

# import models
from SeqModel.transformer import ForcastingTransformer

# Other needs:
from Utils.utils import inference
from Utils.loss_fn import loss_settings
from Utils.plot_fn import plot_prediction_curve, plot_error_heatmap
import Utils.Assistant_func.settings as settings
from Utils.Assistant_func.initialization import Path_init, Display_info

def parse_args():
    parser = argparse.ArgumentParser(description='Training Process of Low-dose Tau PET Enhancement Project')
    parser.add_argument(
        '--key', default=113, dest='trainID', help='key of the result path', type=int)
    parser.add_argument(
        "--null", default='------------------------', dest="NULL")
    args, rest = parser.parse_known_args()
    return args, rest

def main():

    # Initialization----
        # create Config
        # seed: set a seed for reproducibility
        # set decive, path
    args, restargs = parse_args()
    config = Config(args, restargs)
    path = Path_init("path_file.yaml", config.trainID, -1, create_new=False)
    config.import_training_paras(os.path.join(path['result_path'], path['training_cfg']))
    config.import_paras("hyperparameters.yaml")
    Display_info(vars(config), path, create_new=False)
    settings.set_seed(config.seed)
    config.device = settings.set_device(config.device)

    # Dataloader----
        # dataset split, dataset, dataloader
    data_path = path['data_path']

    test_set = M5_Dataset(data_path, Twin=[56,56], Tpred=28, is_inference=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=config.CPU_workers)

    # Testing----
        # best model, tester
    print("\n*\n*\n----START TESTING----\n")
    model_best = ForcastingTransformer(input_dim=3049, store_dim=10, date_dim=11, model_dim=config.transformer_dim, nhead=config.n_heads,\
                                       num_encoder_layers=config.enc_layers, num_decoder_layers=config.dec_layers, dim_feedforward=2048).to(config.device)
    checkpoint = torch.load(path['model_path'], map_location=config.device)
    model_best.load_state_dict(checkpoint['model'])
    criterion = loss_settings(loss_type="MSE")
    print(f"  Model was saved at epoch of {checkpoint['epoch']+1}\n")

    fig_dir = os.path.join(path['result_path'], 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    num_round = 4
    model_best.eval()
    Tpred = config.output_days
    total_loss = 0.0
    with torch.no_grad():
        PRED = np.empty((10, 28, 3049), dtype=np.float32)
        GT = np.empty((10, 28, 3049), dtype=np.float32)
        for idx, datapack in enumerate(test_loader):
            gt, data, enc_date_, store = datapack
            data, gt, enc_date_, store = data.to(config.device), gt.to(config.device), enc_date_.to(config.device), store.to(config.device)
            pred = torch.zeros_like(gt).to(config.device)
            for round in range(num_round):
                enc_date = enc_date_[:,round*Tpred:data.size()[1]+(round+1)*Tpred,...]
                output = model_best(data, store, enc_date, out_seqlen=Tpred)
                pred = torch.cat([pred, output], dim=1)
                data = data[:,Tpred:,...]
                data = torch.cat([data, output], dim=1)
            pred = data[:, -num_round*Tpred:, :]
            pred = pred.round_()
            loss = criterion(pred, gt).detach().item()
            total_loss += loss
            PRED[idx, ...] = pred.detach().cpu().numpy()[0,...]
            GT[idx, ...] = gt.detach().cpu().numpy()[0,...]
            
            item_id = 2973
            plot_prediction_curve(gt.detach().cpu().numpy()[0,:,item_id], pred.detach().cpu().numpy()[0,:,item_id], 1913, title=f"Item{item_id}_{test_set.store_dir[idx]}", save_path=os.path.join(fig_dir, f"Item{item_id}_{test_set.store_dir[idx]}.png"))
            # item_id = 1400
            # plot_prediction_curve(gt.detach().cpu().numpy()[0,:,item_id], pred.detach().cpu().numpy()[0,:,item_id], 1913, title=f"Item{item_id}_{test_set.store_dir[idx]}", save_path=os.path.join(fig_dir, f"Item{item_id}_{test_set.store_dir[idx]}.png"))
            # item_id = 2600
            # plot_prediction_curve(gt.detach().cpu().numpy()[0,:,item_id], pred.detach().cpu().numpy()[0,:,item_id], 1913, title=f"Item{item_id}_{test_set.store_dir[idx]}", save_path=os.path.join(fig_dir, f"Item{item_id}_{test_set.store_dir[idx]}.png"))
        
        print(f"Final RMSE: {np.sqrt(total_loss / len(test_loader.dataset))}")
        plot_error_heatmap(GT, PRED, save_path=os.path.join(fig_dir, f"Heatmap.png"))
            
    print(f"*\n*\n----FINISHED----\n")


if __name__ == "__main__":
    main()