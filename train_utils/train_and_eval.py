import torch
from torch import nn
import train_utils.distributed_utils as utils

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        # Ignore the pixel with a value of 255 in target, which is the target edge or padding.
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            f=[0,0,0,0] 
            n=1
            output, _ = model(image,f,n)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr

def train_one_epoch_add_W(model, optimizer, data_loader, device, epoch, f, n, lr_scheduler,  print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if isinstance(n, list):
        batch_f = []
        batch_n = []
        batch_size = 4
        i = 0
        tmp_f = []
        tmp_n = []
        for a_f, a_n in zip(f,n):
            
            tmp_f.append(a_f)
            tmp_n.append(a_n)
            if i==len(f) - 1:
                batch_f.append(tmp_f)
                batch_n.append(tmp_n)
            if i%batch_size==batch_size - 1 and i!=0:            
                batch_f.append(tmp_f)
                batch_n.append(tmp_n)
                tmp_f = []
                tmp_n = []
            i = i + 1

    all_layer_features=[]
    i = 0
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # output = model(image, f, n)
            if isinstance(n, list):
                output, layer_features = model(image, batch_f[i], batch_n[i])
            else:
                output, layer_features = model(image, f, n)
            all_layer_features.extend(layer_features)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr, all_layer_features
    # return metric_logger.meters["loss"].global_avg, lr

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # The lr magnification factor in the warp process is from warmup_factor -> 1.
            return warmup_factor * (1 - alpha) + alpha
        else:
            # The lr magnification factor after warmup is from 1 -> 0.
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
