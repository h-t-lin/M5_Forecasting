import torch
from tqdm import tqdm

def train(data_loader, model, criterion, optimizer, lr_scheduler, device, eta_L:int, display_text="  Training"):
    model.train()
    total_loss = 0.0
    for gt, data, enc_date, store in tqdm(data_loader, desc=display_text):
        data, gt, enc_date, store = data.to(device), gt.to(device), enc_date.to(device), store.to(device)

        optimizer.zero_grad()
        output = model(data, store, enc_date, out_seqlen=gt.size()[1])
        Dloss = criterion(output, gt)
        MAloss = criterion(output.mean(dim=1), gt.mean(dim=1))
        loss = eta_L * Dloss + (1-eta_L) * MAloss
        loss.backward()
        optimizer.step()

        total_loss += Dloss.detach().item() * gt.size()[0]
    lr_scheduler.step() 
    return total_loss

def inference(data_loader, model, criterion, device, num_round=4, display_text="  Validation"):
    model.eval()
    total_loss = 0.0
    Tpred = data_loader.dataset.Tpred // num_round
    with torch.no_grad():
        for gt, data, enc_date_, store in tqdm(data_loader, desc=display_text):
            data, gt, enc_date_, store = data.to(device), gt.to(device), enc_date_.to(device), store.to(device)
            pred = torch.zeros_like(gt).to(device)
            for round in range(num_round):
                enc_date = enc_date_[:,round*Tpred:data.size()[1]+(round+1)*Tpred,...]
                output = model(data, store, enc_date, out_seqlen=Tpred)
                pred = torch.cat([pred, output], dim=1)
                data = data[:,Tpred:,...]
                data = torch.cat([data, output], dim=1)
            pred = pred[:, num_round*Tpred:, :]
            loss = criterion(pred, gt)
            total_loss += loss.detach().item() * gt.size()[0]
    return total_loss
