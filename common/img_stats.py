import numpy as np
import argparse
import time
from models.cirtorch_utils.genericdataset import ImagesFromList
from evaluate import get_files, time_to_str

parser = argparse.ArgumentParser()
parser.add_argument('--database_dir', default='data/AachenDayNight/images_upright/db', help='Specify where images are stored')
parser.add_argument('--save_file', default='data/img_stats.txt', help='Where to store results')
parser.add_argument('--print_chunk', type=float, default=10, help='Print after every completed nth chunk of dataset')
parser.add_argument('--overfit', type=int, default=None, help='Reduce num images for testing')
args = parser.parse_args()


t = time.time()

img_names = get_files(args.database_dir, '*.jpg')
images = ImagesFromList('', img_names)

ns, ms, ss = [], [], []
for i, img in enumerate(images):
    #print('img shape: {}'.format(img.size))
    if (i % 5) == 0:   #(len(images)//args.print_chunk)) == 0:
        print('\rCompleted {:4d}/{} images'.format(i, len(images)), end = '')
    n_pixel = np.multiply(*img.size)
    mean, std = np.mean(img, axis=(0,1)), np.std(img, axis=(0,1))
    ns.append(n_pixel)
    ms.append(mean)
    ss.append(std)
    #print('n={}\tm={}\tstd={}'.format(n_pixel, mean, std))
    if args.overfit is not None and i > args.overfit:
        break
print('')
ns = np.array(ns)
ms = np.stack(ms)
ss = np.stack(ss)
overall_mean = np.average(ms, axis=0, weights=ns)
## formula is based on https://www.researchgate.net/post/How_to_combine_standard_deviations_for_three_groups
os1 = np.dot(ns, ss**2)
os2 = np.dot(ns, (ms - np.mean(ms, axis=0))**2)
overall_var = (os1 + os2)/np.sum(ns).astype(np.float)
overall_std = np.sqrt(overall_var)


print('Overall mean: {}'.format(overall_mean))
print('Overall std:  {}'.format(overall_std))
result_store = np.vstack((overall_mean, overall_std))
if args.save_file != 'None':
    np.savetxt(args.save_file, result_store)
t = time.time() - t
print('Finished in {}\n({}/image)'.format(time_to_str(t), time_to_str(t/float(len(images)))))
    
