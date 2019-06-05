from visdom import Visdom
import numpy as np

"""
Based on https://github.com/noagarcia/visdom-tutorial
"""
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, server, port, env_name='main'):
        self.viz = Visdom(server=server, port=port)
        self.env = env_name
        self.plots = {}
        
    def add_plot(self, var_name, legend, title_name):
        x, y = 0, 0
        self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=legend,
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            raise KeyError('Plot for %s not defined yet'%var_name)
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')