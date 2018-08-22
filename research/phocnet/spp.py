# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Reference: https://github.com/yhenon/keras-spp/blob/master/spp/SpatialPyramidPooling.py

import tensorflow as tf

class SPP(tf.keras.layers.Layer):
  def __init__(self, pool_list, **kwargs):
    self.dim_ordering = tf.image_dim_ordering()
    assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

    self.pool_list = pool_list

    self.num_outputs_per_channel = sum([i*i for i in pool_list])

    super.(SPP, self).__init__(**kwargs)

  def build(self, input_shape):
    if self.dim_ordering == 'th':
      self.nb_channels = input_shape[1]
    elif self.dim_ordering == 'tf':
      self.nb_channels = input_shape[3]

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.nb_channels*self.num_outputs_per_channel)

  def get_config(self):
    config = {'pool_list': self.pool_list}
    base_config = super(SPP, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, x, mask=None):
  
    input_shape = tf.shape(x)

    if self.dim_ordering == 'th':
      num_rows = input_shape[2]
      num_cols = input_shape[3]
    elif self.dim_ordering == 'tf':
      num_rows = input_shape[1]
      num_cols = input_shape[2]

    row_length = [tf.cast(num_rows, 'float32') / i for i in self.pool_list]
    col_length = [tf.cast(num_cols, 'float32') / i for i in self.pool_list]

    outputs = []

    if self.dim_ordering == 'th':
      for pool_num, num_pool_regions in enumerate(self.pool_list):
        for jy in range(num_pool_regions):
          for ix in range(num_pool_regions):
            x1 = ix * col_length[pool_num]
            x2 = ix * col_length[pool_num] + col_length[pool_num]
            y1 = jy * row_length[pool_num]
            y2 = jy * row_length[pool_num] + row_length[pool_num]

            x1 = tf.cast(tf.round(x1), 'int32')
            x2 = tf.cast(tf.round(x2), 'int32')
            y1 = tf.cast(tf.round(y1), 'int32')
            y2 = tf.cast(tf.round(y2), 'int32')
            new_shape = [input_shape[0], input_shape[1], y2-y1, x2-x1]
            x_crop = x[:, :, y1:y2, x1:x2]
            xm = tf.reshape(x_crop, new_shape)
            pooled_val = tf.max(xm, axis=(2,3))
            outputs.append(pooled_val)

    elif self.dim_ordering == 'tf':
      for pool_num, num_pool_regions in enumerate(self.pool_list):
        for jy in range(num_pool_regions):
          for ix in range(num_pool_regions):
            x1 = ix * col_length[pool_num]
            x2 = ix * col_length[pool_num] + col_length[pool_num]
            y1 = jy * row_length[pool_num]
            y2 = jy * row_length[pool_num] + row_length[pool_num]

            x1 = tf.cast(tf.round(x1), 'int32')
            x2 = tf.cast(tf.round(x2), 'int32')
            y1 = tf.cast(tf.round(y1), 'int32')
            y2 = tf.cast(tf.round(y2), 'int32')

            new_shape = [input_shape[0], y2-y1, x2-x1, input_shape[3]]
            x_crop = x[:, y1:y2, x1:x2, :]
            xm = tf.reshape(x_crop, new_shape)
            pooled_val = tf.max(xm, axis=(1, 2))
            outputs.append(pooled_val)

    if self.dim_ordering == 'th':
      outputs = tf.concatenate(outputs)
    elif self.dim_ordering == 'tf':
      outputs = tf.concatenate(outputs)

    return outputs
