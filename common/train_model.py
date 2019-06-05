import torch
import os
import visdom
import shutil
import numpy as np

from common.utils import VisdomLinePlotter


def train_classifier(model, dataset, config, args, **kwargs):
    ## Init params
    training_params = config['Training']
    logging_params = config['Logging']
    hyper_params = config['Hyperparams']
    
    n_epochs = training_params.getint('n_epochs')
    overfit = training_params.getint('overfit', -1)
    n_samples = len(dataset) if overfit < 0 else overfit
    
    log_division = logging_params.getint('log_division',10)
    log_division = log_division if overfit < 0 else overfit // 2
    save_interval = training_params.getint('save_interval', n_epochs//2)
    do_val = training_params.getboolean('validation', True)
    val_frequency = training_params.getint('val_frequency', 1)
    
    ## Set up logging directory
    base_dir = os.path.join('logs', kwargs['exp_name'])
    i = 0
    while os.path.isdir(base_dir):
        base_dir = os.path.join('logs', kwargs['exp_name'], '_'+str(i))
        i += 1
    print('Logging to %s'%base_dir)
    os.mkdir(base_dir)
    shutil.copyfile(args.config, os.path.join(base_dir, 'config.ini'))
    
    ## Init cuda
    CUDA = torch.cuda.is_available()
    if CUDA:
        print('Cuda available on %d devices'%(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            print(' - %d: %s'%(i, str(torch.cuda.get_device_name(i))))
            
        torch.backends.cudnn.benchmark=True
        
        model = model.cuda()
    if 'device' in kwargs:
        os.environ['CUDA_VISIBLE_DEVICES'] = kwargs['device']
        print('Devices set to %s'%kwargs['device'])
        
    ## Training params
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=hyper_params.getfloat('lr', 1e-3))
    
    ## Setup visdom logging
    use_visdom = logging_params.getboolean('visdom', False)
    if use_visdom:
        plotter = VisdomLinePlotter(
            env_name='Classifier',
            server = 'localhost' if 'hostname' not in kwargs else kwargs['hostname'], 
            port = 8097 if 'port' not in kwargs else kwargs['port'])
        plotter.add_plot('Loss', ['Train loss', 'Val Loss'], 'Loss')
    
    ## Actual training
    model.train()
    for epoch in range(n_epochs):
        print('Epoch %d/%d'%(epoch,n_epochs))
        if epoch % save_interval == 0:
            save_checkpoint(epoch, n_epochs, model, optim, base_dir)
        
        ## Validation
        if do_val and epoch % val_frequency == 0:
            print('Validation in progress')
            model.eval()
            val_losses = []
            n_val = len(kwargs['val_set'])
            for i, (data, target) in enumerate(kwargs['val_set']):
                if CUDA:
                    data = data.cuda()
                    target = target.cuda()
                val_loss = step_feedfwd(model, data.unsqueeze(0), target.unsqueeze(0), loss_fn, None,train=False)
                val_losses.append(val_loss.item())
                if i % (n_val//log_division) == 0:
                    print('\t%d/%d samples (Loss: %.3f)'%(i, n_val, val_loss.item()))
            plotter.plot('Loss', 'Val loss', epoch, np.mean(val_losses))
            model.train()
            
            
        ## Iterate over training data    
        print('Training')
        for i, (data, target) in enumerate(dataset):
            if i > n_samples: ##overfit
                break
            if CUDA:
                data = data.cuda()
                target = target.cuda()
            loss = step_feedfwd(model, data, target, loss_fn, optim)
                
                
            ## Logging 
            if i % (n_samples//log_division) == 0:
                
                print('\t%d/%d samples (Loss: %.3f)'%(i, n_samples, loss.item()))
                if use_visdom:
                    plotter.plot('Loss', 'Train loss', float(epoch) + float(i)/float(n_samples), loss.item())
                    
            
                    
    ## Save training
    save_checkpoint(epoch, n_epochs, model, optim, base_dir)
            
def step_feedfwd(model, data, target, loss_fn, optim, train=True):
    if train:
        optim.zero_grad()
    prediction = model(data)
    loss = loss_fn(prediction, target)
    if train:
        loss.backward()
        optim.step()
    return loss

def save_checkpoint(epoch, final_epoch, model, optim, save_dir):
    if epoch >= final_epoch:
        filename = os.path.join(save_dir, 'final_model.pth.tar'.format(epoch))
    else:
        filename = os.path.join(save_dir, 'epoch_{:03d}.pth.tar'.format(epoch))
    checkpoint_dict =\
            {'epoch': epoch, 'model_state_dict': model.state_dict(),
            # 'criterion_state_dict': self.train_criterion.state_dict()
            }
    print('Save model to %s'%filename)
    torch.save(checkpoint_dict, filename)