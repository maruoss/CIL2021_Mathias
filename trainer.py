import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from trainer_visualizer import show_val_samples
from torch.utils.tensorboard import SummaryWriter
import copy

def train_model(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, device, n_epochs):
    # training loop
    # logdir = 'tensorboard/100dice.lr0.001.batch8.img400.ep50'
    writer = SummaryWriter()  # tensorboard writer (can also log images)
    since = time.time()

    history = {}  # collects metrics at the end of each epoch
    best_val_loss = 1e10 # Initialize best val loss
    best_model_wts = copy.deepcopy(model.state_dict()) # Initialize best model weights
    best_epoch = 1 # Initialize at which epoch best model is saved

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

            # Add transformed train image to tensorboard
            # x0 = x[0].detach()
            # writer.add_image("train_image_transformed", x0, epoch)

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
        for k, v in history[epoch].items():
          writer.add_scalar(k, v, epoch) # log to tensorboard
        # writer.close() #close writer
        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()])) # print epoch losses/ metrics
        # CAUTION: Below function causes Memory Leakage if run on a lot of epochs! (not solved yet):
        # show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()) # show val samples and predicted masks

        # Save model weights if best val loss in epoch:
        if history[epoch]["val_loss"] < best_val_loss:
            best_val_loss = history[epoch]["val_loss"]
            best_model_wts = copy.deepcopy(model.state_dict()) # deepcopy otherwise its just referenced and saves overfitted model instead, recommended on PyTorch website.
            best_epoch = epoch+1 # epoch starts at 0

        # Add images to tensorboard, as eval dataloader is shuffled x will be different each instance
        # x0, x1, y0, y1, y_hat0, y_hat1 = x[0], x[1], y[0], y[1], y_hat[0], y_hat[1]
        # writer.add_image("val_image", x0, epoch)
        # writer.add_image("val_groundtruth", y0, epoch)
        # writer.add_image("predicted_mask", y_hat0, epoch)
        to_visualize = 2
        to_stack = []
        for i in range(min(to_visualize, len(x))):
            to_stack += [x[i].detach(), y[i].detach().repeat(3, 1, 1), y_hat[i].detach().repeat(3, 1, 1)] #detach, but still on GPU?
        writer.add_images("val_image, val_groundtruth, prediction_mask", torch.stack(to_stack, dim=0), epoch)
        # writer.add_images("val_image, val_groundtruth, prediction_mask", 
        # torch.stack((x0, y0.repeat(3, 1, 1), y_hat0.repeat(3, 1, 1)), dim=0), epoch) # repeat y's with shape (1, H, W) along dim 0 -> (3, H, W)
        writer.close()
 



    print('Finished Training')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f"Total Epochs run: {n_epochs}")
    print('Best validation loss: {:.4f} after {} epochs'.format(best_val_loss, best_epoch))
    print(f"Model returned after {best_epoch} epochs")

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

    # load best model weights (lowest val loss in epochs)
    model.load_state_dict(best_model_wts)
    return model