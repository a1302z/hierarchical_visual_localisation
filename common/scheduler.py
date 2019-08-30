import numpy as np


"""
Available schedule plans:
log_linear : Linear interpolation with log learning rate scale
log_cosine : Cosine interpolation with log learning rate scale
"""
class LearningRateScheduler():
    def __init__(self, total_epochs, log_start_lr, log_end_lr, schedule_plan='log_linear'):
        self.total_epochs = total_epochs
        if schedule_plan == 'log_linear':
            self.calc_lr = lambda epoch: np.power(10, ((log_end_lr-log_start_lr)/total_epochs)*epoch + log_start_lr)
        elif schedule_plan == 'log_cosine':
            self.calc_lr = lambda epoch: np.power(10, (np.cos(np.pi*(epoch/total_epochs))/2.+.5)*abs(log_start_lr-log_end_lr) + log_end_lr)
        else:
            raise NotImplementedError('Requested learning rate schedule {} not implemented'.format(schedule_plan))
            
            
    def get_lr(self, epoch):
        if (type(epoch) is int and epoch > self.total_epochs) or (type(epoch) is np.ndarray and np.max(epoch) > self.total_epochs):
            raise AssertionError('Requested epoch out of precalculated schedule')
        return self.calc_lr(epoch)
    
    def adjust_learning_rate(self, optimizer, epoch):
        new_lr = self.get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    epochs = 10
    scheduler_linear = LearningRateScheduler(epochs, -2, -4, schedule_plan = 'log_linear')
    scheduler_cosine = LearningRateScheduler(epochs, -2, -4, schedule_plan = 'log_cosine')
    num_points = 1000. # number of points for plot
    mult_factor = num_points/epochs 
    x = np.arange(epochs*mult_factor)/mult_factor
    lin_lr = scheduler_linear.get_lr(x)
    cos_lr = scheduler_cosine.get_lr(x)
    #plt.yscale('log')
    plt.plot(x, lin_lr, label='linear log schedule')#, s=0.5)
    plt.plot(x, cos_lr, label='cosine log schedule')#, s=0.5)
    #plt.plot(x, np.abs(lin_lr-cos_lr), label='difference')#, s=0.5)
    plt.legend()
    plt.savefig('figures/schedule_linear_space.png')
    plt.show()