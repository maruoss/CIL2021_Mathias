import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from trainer_visualizer import show_val_samples

def train_epoch(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, device, n_epochs):
    # training loop
    # logdir = './tensorboard/net'
    # writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)
    since = time.time()

    history = {}  # collects metrics at the end of each epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        # training
        model.train()
        for (x, y) in pbar:
            optimizer.zero_grad()  # zero out gradients
            y = y.to(device)
            x = x.to(device) # add to device here -> faster than in Dataset itself!
            y_hat = model(x)["out"]  # forward pass #MATHIAS: need ["out"] for deeplabv3
            # # ADJUST: Round groundtruth to 0, 1? 
            # y = (y > CUTOFF).float() ###################################################### 0, 1 TARGET rounded on CUTOFF
            loss = loss_fn(y_hat, y)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights

            # log partial metrics
            metrics['loss'].append(loss.item())
            assert str(loss_fn) in ("BCEWithLogitsLoss()", "BinaryDiceLoss_Logits()"), "no logit loss function used" # Otherwise, torch sigmoid is not necessary here, e.g. with BCELoss  
            y_hat = torch.sigmoid(y_hat) # For metrics, torch.sigmoid needed!
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item()) # TORCH SIGMOID HERE AS WELL -> LIKE A PREDICTION
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                x = x.to(device) # add to device here -> faster than in Dataset itself!
                y = y.to(device)
                # logits of pixel being 0 or 1:
                y_hat = model(x)["out"] # forward pass #MATHIAS: added "out". removed torch.sigmoid -> logits are needed for loss_fn
                # # ADJUST: Round groundtruth to 0, 1? 
                # y = (y > CUTOFF).float() ###################################################### 0, 1 TARGET rounded on CUTOFF 
                loss = loss_fn(y_hat, y)
                
                # log partial metrics
                metrics['val_loss'].append(loss.item())
                assert str(loss_fn) in ("BCEWithLogitsLoss()", "BinaryDiceLoss_Logits()"), "no logit loss function used"  
                y_hat = torch.sigmoid(y_hat) # For metrics, torch.sigmoid needed!
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        # for k, v in history[epoch].items():
        #   writer.add_scalar(k, v, epoch)
        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))
        show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

    print('Finished Training')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Show plot for losses
    plt.plot([v['loss'] for k, v in history.items()], label='Training Loss')
    plt.plot([v['val_loss'] for k, v in history.items()], label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    # Show plots for all additional metrics # UNCOMMENT/ COMMENT OUT
    # for k, _ in metric_fns.items():
    #     plt.plot([v[k] for _, v in history.items()], label='Training '+k)
    #     plt.plot([v["val_"+k] for _, v in history.items()], label='Validation '+k)
    #     plt.ylabel(k)
    #     plt.xlabel('Epochs')
    #     plt.legend()
    #     plt.show()