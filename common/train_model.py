import torch
import os
import visdom
import shutil
import numpy as np
import time
from datetime import timedelta

from common.utils import VisdomLinePlotter

from torch_geometric.data import Data


def train_classifier(ptnet, cnn2d, dataset, config, args, **kwargs):
    ## Init params
    training_params = config['Training']
    logging_params = config['Logging']
    hyper_params = config['Hyperparams']
    
    n_epochs = training_params.getint('n_epochs')
    overfit = training_params.getint('overfit', -1)
    n_samples = len(dataset) if overfit < 0 else overfit // config['Training'].getint('batch_size', 1)
    
    log_division = logging_params.getint('log_division',10)
    log_division = log_division if overfit < 0 else 1
    save_interval = training_params.getint('save_interval', n_epochs//2)
    do_val = training_params.getboolean('validation', True)
    val_frequency = training_params.getint('val_frequency', 1)
    
    #seed all randomness
    seed = training_params.getint('random_seed')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    ## Set up logging directory
    exp_name = kwargs['exp_name']
    base_dir = os.path.join('logs', exp_name)
    i = 1
    while os.path.isdir(base_dir):
        exp_name = kwargs['exp_name']+ '_'+str(i)
        base_dir = os.path.join('logs', exp_name)
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
        
        ptnet = ptnet.cuda()
        cnn2d = cnn2d.cuda()
        device = torch.device('cuda')
    if 'device' in kwargs:
        os.environ['CUDA_VISIBLE_DEVICES'] = kwargs['device']
        print('Devices set to %s'%kwargs['device'])
        
    ## Training params
    #loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.TripletMarginLoss(margin=config['Hyperparams'].getfloat('loss_margin', 1.0))
    optim = torch.optim.Adam(list(ptnet.parameters()) + list(cnn2d.parameters()), lr=hyper_params.getfloat('lr', 1e-3))
    
    ## Setup visdom logging
    use_visdom = logging_params.getboolean('visdom', False)
    if use_visdom:
        plotter = VisdomLinePlotter(
            env_name='Triplet/'+exp_name,
            server = 'localhost' if 'hostname' not in kwargs else kwargs['hostname'], 
            port = 8097 if 'port' not in kwargs else kwargs['port'])
        plotter.add_plot('Loss', ['Train loss', 'Val Loss'], 'Loss')
    
    ## Actual training
    ptnet.train()
    cnn2d.train()
    for epoch in range(n_epochs):
        print('Epoch %d/%d'%(epoch,n_epochs))
        if epoch % save_interval == 0:
            save_checkpoint(epoch, n_epochs, [ptnet, cnn2d], optim, base_dir)
        
        ## Validation
        if do_val and (epoch % val_frequency) == 0:
            print('Validation in progress')
            t1 = time.time()
            ptnet.eval()
            cnn2d.eval()
            val_losses = []
            n_val = len(kwargs['val_set']) if overfit <= 0 else overfit
            for i, data in enumerate(kwargs['val_set']):
                if i > n_val:
                    break
                if CUDA:
                    data[0] = data[0].cuda()
                    data[1] = data[1].to(device)#Data(pos=data[1].cuda(), batch=torch.cuda.LongTensor([0]*data[1].size(0)))
                    data[2] = data[2].to(device)#Data(pos=data[2].cuda(), batch=torch.cuda.LongTensor([0]*data[2].size(0)))
                #else:
                #    data[1] = Data(pos=data[1], batch=torch.LongTensor([0]*data[1].size(0)))
                #    data[2] = Data(pos=data[2], batch=torch.LongTensor([0]*data[2].size(0)))
                with torch.no_grad():
                    val_loss = step_feedfwd_triplet(ptnet, cnn2d, data, loss_fn, None, train=False)    
                #val_loss = step_feedfwd(model, data.unsqueeze(0), target.unsqueeze(0), loss_fn, None,train=False)
                val_losses.append(val_loss.item())
                if i % (n_val//log_division) == 0:
                    print('\t%d/%d samples (Loss: %.3f)'%(i, n_val, np.mean(val_losses)))
            plotter.plot('Loss', 'Val loss', epoch, np.mean(val_losses))
            ptnet.train()
            cnn2d.train()
            t2 = time.time()
            print('\tValidation time: %s'%str(timedelta(seconds=t2-t1)))
            
            
        ## Iterate over training data    
        print('Training')
        t1 = time.time()
        losses = []
        for i, data in enumerate(dataset):
            if i > n_samples: ##overfit
                break
            if CUDA:
                data[0] = data[0].cuda()
                data[1] = data[1].to(device)#Data(pos=data[1].cuda(), batch=torch.cuda.LongTensor([0]*data[1].size(0)))
                data[2] = data[2].to(device)#Data(pos=data[2].cuda(), batch=torch.cuda.LongTensor([0]*data[2].size(0)))
            #else:
            #    data[1] = Data(pos=data[1], batch=torch.LongTensor([0]*data[1].size(0)))
            #    data[2] = Data(pos=data[2], batch=torch.LongTensor([0]*data[2].size(0)))
                
            #loss = step_feedfwd(model, data, target, loss_fn, optim)
            loss = step_feedfwd_triplet(ptnet, cnn2d, data, loss_fn, optim)    
                
            losses.append(loss.item())
            ## Logging 
            if i % (n_samples//log_division) == 0 and (epoch == 0 or i > 0):
                m_loss = np.mean(losses)
                print('\t%d/%d samples (Mean loss: %.3f)'%(i, n_samples, m_loss))
                if use_visdom:
                    plotter.plot('Loss', 'Train loss', float(epoch) + float(i)/float(n_samples), m_loss)
                losses = []
        t2 = time.time()
        print('\tTraining time: %s'%str(timedelta(seconds=t2-t1)))
                    
    ## Eval model result        
    if do_val and epoch % val_frequency == 0:
        print('Validation in progress')
        t1 = time.time()
        ptnet.eval()
        cnn2d.eval()
        val_losses = []
        correct = 0
        n_val = len(kwargs['val_set']) if overfit <= 0 else overfit
        for i, data in enumerate(kwargs['val_set']):
            if i > n_val:
                break
            if CUDA:
                data[0] = data[0].cuda()
                data[1] = data[1].to(device)#Data(pos=data[1].cuda(), batch=torch.cuda.LongTensor([0]*data[1].size(0)))
                data[2] = data[2].to(device)#Data(pos=data[2].cuda(), batch=torch.cuda.LongTensor([0]*data[2].size(0)))
            #else:
            #    data[1] = Data(pos=data[1], batch=torch.LongTensor([0]*data[1].size(0)))
            #    data[2] = Data(pos=data[2], batch=torch.LongTensor([0]*data[2].size(0)))
            with torch.no_grad():
                val_loss = step_feedfwd_triplet(ptnet, cnn2d, data, loss_fn, None, train=False) 
            val_losses.append(val_loss.item())
            #if prediction.argmax() == target.item():
            #    correct += 1
            if i % (n_val//log_division) == 0:
                print('\t%d/%d samples (Loss: %.3f)'%(i, n_val, val_loss.item()))
            plotter.plot('Loss', 'Val loss', n_epochs, np.mean(val_losses))
        ptnet.train()
        cnn2d.train()
        t2 = time.time()
        print('\tValidation time: %s'%str(timedelta(seconds=t2-t1)))
        #print('Final performance: %d/%d (%.1f%%) correctly classified'%(correct, n_val, 100.0*(float(correct)/float(n_val))))
    
    ## Save training
    save_checkpoint(epoch, n_epochs, [ptnet, cnn2d], optim, base_dir)
            
def step_feedfwd(model, data, target, loss_fn, optim, train=True):
    if train:
        optim.zero_grad()
    prediction = model(data)
    loss = loss_fn(prediction, target)
    if train:
        loss.backward()
        optim.step()
    return loss

def step_feedfwd_triplet(ptnet, cnn2d, data, loss_fn, optim, train=True):
    if train:
        optim.zero_grad()
    if len(data[0].size()) <= 3:
        data[0] = data[0].unsqueeze(0)
    anchor = cnn2d(data[0]).transpose(0,1)
    #Data(pos=torch.Tensor(10, 3), batch=torch.LongTensor([0]*10))
    pos = ptnet(data[1])
    neg = ptnet(data[2])
    #print('Shape anchor {}'.format(anchor.size()))
    #print('Shape pos {}'.format(pos.size()))
    #print('Shape neg {}'.format(neg.size()))
    loss = loss_fn(anchor, pos, neg)
    if train:
        loss.backward()
        optim.step()
    return loss

def save_checkpoint(epoch, final_epoch, models, optim, save_dir):
    if epoch >= final_epoch:
        filename1 = os.path.join(save_dir, 'final_model_ptnet.pth.tar'.format(epoch))
        filename2 = os.path.join(save_dir, 'final_model_cnn2d.pth.tar'.format(epoch))
    else:
        filename1 = os.path.join(save_dir, 'ptnet_epoch_{:03d}.pth.tar'.format(epoch))
        filename2 = os.path.join(save_dir, 'cnn2d_epoch_{:03d}.pth.tar'.format(epoch))
    checkpoint_dict1 =\
            {'epoch': epoch, 'model_state_dict': models[0].state_dict(),
            # 'criterion_state_dict': self.train_criterion.state_dict()
            }
    checkpoint_dict2 =\
            {'epoch': epoch, 'model_state_dict': models[1].state_dict(),
            # 'criterion_state_dict': self.train_criterion.state_dict()
            }
    print('Save ptnet to %s'%filename1)
    print('Save cnn2d to %s'%filename2)
    torch.save(checkpoint_dict1, filename1)
    torch.save(checkpoint_dict2, filename2)
    