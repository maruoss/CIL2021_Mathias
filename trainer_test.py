import time
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from trainer_visualizer import show_val_samples
from torch.utils.tensorboard import SummaryWriter
import copy
from test_augmentation import test_augmentation
from patch_test_augmentation import patch_test_augmentation

# THIS IS TRAINING ON FULL TRAINING SET FOR FINAL SUBM. TO KAGGLE

def train_test(train_dataloader, model, loss_fn, metric_fns, optimizer, device, n_epochs, comment:str):
    # training loop
    # logdir = 'tensorboard/100dice.lr0.001.batch8.img400.ep50'
    writer = SummaryWriter(comment=comment)  # tensorboard writer (can also log images)
    since = time.time()

    history = {}  # collects metrics at the end of each epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        # training
        model.train()
        for (x, y) in pbar:
            optimizer.zero_grad()  # zero out gradients
            y = y.to(device)
            x = x.to(device) # add to device here -> faster than in Dataset itself!
            # y_hat = model(x)["out"]  # forward pass #MATHIAS: need ["out"] for deeplabv3
            # # ADJUST: Round groundtruth to 0, 1? 
            # y = (y > CUTOFF).float() ###################################################### 0, 1 TARGET rounded on CUTOFF
            # loss = loss_fn(y_hat, y)
            model_output = model(x) # Save ordered dict with outputs of classifier and aux classifier
            y_hat = model_output["out"] # access output of main classifier
            aux_output = model_output["aux"] # access output of aux classifier, only relevant for finetuning, doesnt seem to change a lot though
            loss = loss_fn(y_hat, y) + 0.4 * loss_fn(aux_output, y)
            # Patch loss (commented out, seems to not influence training?)
            # y_hat_patched = nn.functional.avg_pool2d(y_hat, 16)
            # y_patched = nn.functional.avg_pool2d(y, 16)
            # loss2 = loss_fn(y_hat_patched, y_patched)
            ##
            # loss = 0.5 * loss1 + 0.5 * loss2 # Pixel loss and patch loss
            loss.backward()  # backward pass
            optimizer.step()  # take gradient step, optimize weights

            # log partial metrics
            metrics['loss'].append(loss.item())
            # assert str(loss_fn) in ("BCEWithLogitsLoss()", "BinaryDiceLoss_Logits()", "BCEDiceLoss_Logits()"), "no logit loss function used" # Otherwise, torch sigmoid is not necessary here, e.g. with BCELoss  
            y_hat = torch.sigmoid(y_hat) # For metrics, torch.sigmoid needed!
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item()) # TORCH SIGMOID HERE AS WELL -> LIKE A PREDICTION
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

            # Add transformed train image to tensorboard
            # x0 = x[0].detach()
            # writer.add_image("train_image_transformed", x0, epoch)

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
          writer.add_scalar(k, v, epoch) # log to tensorboard
        # writer.close() #close writer
        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()])) # print epoch losses/ metrics


        # Add images to tensorboard, as eval dataloader is shuffled x will be different each instance
        # x0, x1, y0, y1, y_hat0, y_hat1 = x[0], x[1], y[0], y[1], y_hat[0], y_hat[1]
        # writer.add_image("val_image", x0, epoch)
        # writer.add_image("val_groundtruth", y0, epoch)
        # writer.add_image("predicted_mask", y_hat0, epoch)
        to_visualize = 2
        to_stack = []
        for i in range(min(to_visualize, len(x))):
            to_stack += [x[i].detach(), y[i].detach().repeat(3, 1, 1), y_hat[i].detach().repeat(3, 1, 1)] #detach, but still on GPU?
        writer.add_images("train_image, train_groundtruth, prediction_mask", torch.stack(to_stack, dim=0), epoch)
        # writer.add_images("val_image, val_groundtruth, prediction_mask", 
        # torch.stack((x0, y0.repeat(3, 1, 1), y_hat0.repeat(3, 1, 1)), dim=0), epoch) # repeat y's with shape (1, H, W) along dim 0 -> (3, H, W)
        writer.close()
 

    print('Finished Training')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f"Total Epochs run: {n_epochs}")


    # Show plot for losses
    # plt.plot([v['loss'] for k, v in history.items()], label='Training Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epochs')
    # plt.legend()
    # plt.show()

