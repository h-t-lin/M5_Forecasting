# Reading/Writing Data
import os
# For plotting
import numpy as np
import matplotlib.pyplot as plt
import seaborn
try: import Utils.Assistant_func.SnL as SnL
except: import Assistant_func.SnL as SnL

def plot_error_heatmap(gt, pred, save_path):
    """
    Plot a heatmap to visualize the prediction error per store and product.
    
    Parameters:
    gt (numpy.ndarray): Ground truth sales data, shape (10, 28, 3000).
    pred (numpy.ndarray): Predicted sales data, shape (10, 28, 3000).
    """
    # 確保資料形狀正確
    gt, pred = np.asarray(gt), np.asarray(pred)
    assert gt.shape == pred.shape

    # 計算28天內的均方誤差
    rmse = np.sqrt(np.mean((gt - pred) ** 2, axis=1))  # shape will be (10, 3000)
    rmse = np.clip(rmse, 0.0, 10.0)

    # 使用seaborn繪製誤差熱力圖
    fig = plt.figure(figsize=(16, 8))
    seaborn.heatmap(rmse, cmap="YlOrRd", cbar_kws={'label': 'Root Mean Squared Error'}, xticklabels=100, yticklabels=range(1, 11))
    
    # 加入標籤和標題
    plt.xlabel("Product ID")
    plt.ylabel("Store ID")
    plt.title("Prediction Error Heatmap (RMSE)")
    fig.savefig(save_path)
    plt.close(fig)

def plot_prediction_curve(gt, pred, starting_day, title="", save_path=""):
    gt, pred = np.asarray(gt), np.asarray(pred)
    assert gt.shape == pred.shape
    days = np.arange(starting_day, starting_day + gt.shape[0])

    fig = plt.figure(figsize=(10, 5))
    plt.plot(days, gt, color='green', linestyle='-', label='Ground Truth')
    plt.plot(days, pred, color='orange', linestyle='-', label='Prediction')
    
    plt.xlabel("days")
    plt.ylabel("sales volume")
    plt.legend()
    plt.title(title)
    fig.savefig(save_path)
    plt.close(fig)
    

def plot_loss_curve(loss_record:dict, epoch, result_path):
    x = np.linspace(1, epoch, epoch)
    plt.plot(x, loss_record["train"], 'g', linestyle='--', label="Training loss")
    plt.plot(x, loss_record["valid"], 'r', linestyle='-.', label="Validation loss")
    # mark the minimum point
    min_value = min(loss_record["valid"])
    min_index = loss_record["valid"].index(min_value)
    middle_point = 0.3*min_value+0.7*max(loss_record["valid"])
    plt.plot(min_index+1, min_value, 'black', linestyle=':', marker='^', markersize=8, alpha=0.7)
    plt.annotate(f'{min_value:.3e}', xy=(min_index+1, min_value), xytext=(min_index+1, middle_point), 
                ha='center', va='baseline', color=(.3,.3,.3), 
                arrowprops=dict(color=(.35,.35,.35), shrink=0.05, width=1, headwidth=6),)

    plt.legend()
    plt.xlabel("Epochs")
    # plt.yscale("log")
    plt.ylabel("Loss")
    plt.suptitle("Loss Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "training_loss_curve.png"))
    plt.clf()
    # save .yaml
    _save_yaml(loss_record['train'], os.path.join(result_path, f'Loss_training.yaml'), start_index=1)
    _save_yaml(loss_record['valid'], os.path.join(result_path, f'Loss_validation.yaml'), start_index=1)
    return

def _save_yaml(indata, save_path, start_index=0):
    if type(indata) is list:
        outdata = {}
        for idx, ele in enumerate(indata):
            outdata[idx+start_index] = ele
    elif type(indata) is dict:
        outdata = indata
    else:
        raise TypeError('datatype must be list or dict')
    SnL.save_yaml(outdata, save_path)

if __name__ == "__main__":
    l1 = [x for x in range(0,20)]
    l2 = [0 for _ in range(8)]
    gt = l2.copy()
    pred = l1.copy()
    gt.extend(l1)
    pred.extend(l2)
    gt = np.array(gt)
    pred = np.array(pred)
    plot_prediction_curve(gt, pred, 1900)