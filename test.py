from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import traceback


import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl

import data
import models
import matplotlib.pyplot as plt


# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', dest='experiment_name', help='experiment_name')
parser.add_argument('--test_int', dest='test_int', type=float, default=1.0, help='test_int')
parser.add_argument('--source', dest='source', help='source')
args_ = parser.parse_args()
with open('./output/%s/setting.txt' % args_.experiment_name) as f:
    args = json.load(f)

# model
atts = args['atts']
n_att = len(atts)
img_size = args['img_size']
shortcut_layers = args['shortcut_layers']
inject_layers = args['inject_layers']
enc_dim = args['enc_dim']
dec_dim = args['dec_dim']
dis_dim = args['dis_dim']
dis_fc_dim = args['dis_fc_dim']
enc_layers = args['enc_layers']
dec_layers = args['dec_layers']
dis_layers = args['dis_layers']
# testing
thres_int = args['thres_int']
test_int = args_.test_int
source = args_.source
# others
use_cropped_img = args['use_cropped_img']
experiment_name = args_.experiment_name


# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================

# data
sess = tl.session()
if source == 'Celeba':
    te_data = data.Celeba('./data', atts, img_size, 1, part='test', sess=sess, crop=not use_cropped_img)

elif source == 'Custom':
    te_data = data.Custom('./data', atts, img_size, 1, part='test', sess=sess, crop=not use_cropped_img)
else:
    print('Incorrect Source: Choose Celeba or Custom')
    exit()

# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers, inject_layers=inject_layers)

# inputs
xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

# sample
x_sample = Gdec(Genc(xa_sample, is_training=False), _b_sample, is_training=False)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
try:
    tl.load_checkpoint(ckpt_dir, sess)
except:
    raise Exception(' [*] No checkpoint!')

# sample
try:
    for idx, batch in enumerate(te_data):
        xa_sample_ipt = batch[0]
        a_sample_ipt = batch[1]
        b_sample_ipt_list = [a_sample_ipt]  # the first is for reconstruction
        for i in range(len(atts)):
            tmp = np.array(a_sample_ipt, copy=True)
            tmp[:, i] = 1 - tmp[:, i]   # inverse attribute

            if source == 'Celeba':
                tmp = data.Celeba.check_attribute_conflict(tmp, atts[i], atts)
            elif source == 'Custom':
                tmp = data.Custom.check_attribute_conflict(tmp, atts[i], atts)
            b_sample_ipt_list.append(tmp)

        x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]

        for i, b_sample_ipt in enumerate(b_sample_ipt_list):
            _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
            if i > 0:   # i == 0 is for reconstruction
                _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int / thres_int
            x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt, _b_sample: _b_sample_ipt}))
        #x_sample_opt_list[2].shape
        #plt.imshow((x_sample_opt_list[2][0]+1)/2)
        #plt.show()
        sample = np.concatenate(x_sample_opt_list, 2)
        #print(sample.shape)
        #print(sample.max())
        #print(sample.min())
        #plt.imshow((sample[0]+1)/2)
        #plt.show()
        save_dir = './output/%s/sample_testing' % experiment_name
        pylib.mkdir(save_dir)
        if source == 'Celeba':
            add = 182638
        else:
            add = 0
        im.imwrite(sample.squeeze(0), '%s/%d.png' % (save_dir, idx+add))

        print('%d.png done!' % (idx+add))

except:
    traceback.print_exc()
finally:
    sess.close()
