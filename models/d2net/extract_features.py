import argparse

import numpy as np

import imageio

import torch

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

import sys
sys.path.insert(0, 'models/d2net/')

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale


# Argument parsing
"""parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    '--image_list_file', type=str, required=True,
    help='path to a file containing a list of images to process'
)

parser.add_argument(
    '--preprocessing', type=str, default='caffe',
    help='image preprocessing (caffe or torch)'
)
parser.add_argument(
    '--model_file', type=str, default='models/d2_tf.pth',
    help='path to the full model'
)

parser.add_argument(
    '--max_edge', type=int, default=1600,
    help='maximum image size at network input'
)
parser.add_argument(
    '--max_sum_edges', type=int, default=2800,
    help='maximum sum of image sizes at network input'
)

parser.add_argument(
    '--output_extension', type=str, default='.d2-net',
    help='extension for the output'
)
parser.add_argument(
    '--output_type', type=str, default='npz',
    help='output file type (npz or mat)'
)

parser.add_argument(
    '--multiscale', dest='multiscale', action='store_true',
    help='extract multiscale features'
)
parser.set_defaults(multiscale=False)

parser.add_argument(
    '--no-relu', dest='use_relu', action='store_false',
    help='remove ReLU after the dense feature extraction module'
)
parser.set_defaults(use_relu=True)

args = parser.parse_args()
"""

class d2net_interface:
    
    def __init__(self, model_file='models/d2_tf.pth', max_edge=1600, max_sum_edges=2800,
                 use_relu=False, use_cuda = torch.cuda.is_available()):
        # CUDA
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # Creating CNN model
        self.model = D2Net(
            model_file=model_file,
            use_relu= use_relu,
            use_cuda=use_cuda
        )
        self.max_edge = max_edge
        self.max_sum_edges = max_sum_edges
        
    def __resize_image__(self, image):
        # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
        resized_image = image
        if max(resized_image.shape) >  self.max_edge:
            resized_image = scipy.misc.imresize(
                resized_image,
                 self.max_edge / max(resized_image.shape)
            ).astype('float')
        if sum(resized_image.shape[: 2]) >  self.max_sum_edges:
            resized_image = scipy.misc.imresize(
                resized_image,
                 self.max_sum_edges / sum(resized_image.shape[: 2])
            ).astype('float')
        return resized_image
    
    
    """
    Tried to adapt process_multiscale method from lib/pyramid.py to work with given keypoints
    """
    def get_features(self, image_path, keypoints, preprocessing='caffe'):
        keypoints_working_copy = keypoints.copy()
        image = imageio.imread(image_path)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)
        resized_image = self.__resize_image__(image)
        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]
        
        #print('{}, {}'.format(fact_i, fact_j))

        input_image = preprocess_image(
             resized_image,
             preprocessing= preprocessing
        )
        with torch.no_grad():
            cur_image = torch.tensor(
                            input_image[np.newaxis, :, :, :].astype(np.float32),
                            device=self.device
                        )
            #print(cur_image.size())
            _, _, h_img, w_img = cur_image.size()
            dense_features = self.model.dense_feature_extraction(cur_image)
            _, _, h, w = dense_features.size()
            #print(dense_features.shape)
            factor_h = float(h_img) / h
            factor_w = float(w_img) / w
            #print('{}, {}'.format(factor_h, factor_w))
            #print(keypoints_working_copy.max(axis=0))
            keypoints_working_copy[:,0] /= factor_h
            keypoints_working_copy[:,1] /= factor_w
            keypoints_working_copy = keypoints_working_copy.astype(np.int32)
            #print(keypoints_working_copy.max(axis=0))
            #print(keypoints_working_copy.shape)
            descriptors = dense_features.cpu().numpy()[0, :, keypoints_working_copy[:,0], keypoints_working_copy[:,1]]
            #print(descriptors.shape)
            """keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=self.device
                ),
                self.model,
                scales=[1], 
                preset_keypoints=keypoints
            )"""
        return descriptors
            
        
        
        
    def extract_features(self, image_list, only_path=True, #only path means that the path to the images is given opposed to image files are given
                         preprocessing='caffe', 
                        output_extension='.d2-net', output_type='npz',
                         multiscale=False, store_results=False):

        #print(args)
        if type(image_list) is not list:
            image_list = [image_list]
        # Process the file
        #for image in tqdm(image_list, total=len(image_list)):
        k, d, s = [], [], []
        for image in image_list:
            if only_path:
                image = imageio.imread(image)
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
                image = np.repeat(image, 3, -1)
                
            resized_image = self.__resize_image__(image)

            

            fact_i = image.shape[0] / resized_image.shape[0]
            fact_j = image.shape[1] / resized_image.shape[1]

            input_image = preprocess_image(
                resized_image,
                preprocessing= preprocessing
            )
            with torch.no_grad():
                if multiscale:
                    keypoints, scores, descriptors = process_multiscale(
                        torch.tensor(
                            input_image[np.newaxis, :, :, :].astype(np.float32),
                            device=self.device
                        ),
                        self.model
                    )
                else:
                    keypoints, scores, descriptors = process_multiscale(
                        torch.tensor(
                            input_image[np.newaxis, :, :, :].astype(np.float32),
                            device=self.device
                        ),
                        self.model,
                        scales=[1]
                    )

            # Input image coordinates
            keypoints[:, 0] *= fact_i
            keypoints[:, 1] *= fact_j
            # i, j -> u, v
            keypoints = keypoints[:, [1, 0, 2]]

            if store_results:
                if  output_type == 'npz':
                    with open(path +  output_extension, 'wb') as output_file:
                        np.savez(
                            output_file,
                            keypoints=keypoints,
                            scores=scores,
                            descriptors=descriptors
                        )
                elif  output_type == 'mat':
                    with open(path +  output_extension, 'wb') as output_file:
                        scipy.io.savemat(
                            output_file,
                            {
                                'keypoints': keypoints,
                                'scores': scores,
                                'descriptors': descriptors
                            }
                        )
                else:
                    raise ValueError('Unknown output type.')
            else:
                k.append(keypoints)
                d.append(descriptors)
                s.append(scores)
        return k, d, s

            
if __name__ == '__main__':
    net = d2net_interface(model_file='data/teacher_models/d2net/d2_tf.pth')
    image = imageio.imread('data/AachenDayNight/images_upright/db/1.jpg')
    keypoints = np.array([[0.5, 0.5], [image.shape[0]-1, image.shape[1]-1], [10, 10]])
    net.get_features('data/AachenDayNight/images_upright/db/1.jpg', keypoints)