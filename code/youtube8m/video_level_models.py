# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

TF_CPP_MIN_LOG_LEVEL=2
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_float(
    "moe_l2", 1e-8,
    "L2 penalty for MoeModel.")
flags.DEFINE_integer(
    "moe_low_rank_gating", -1,
    "Low rank gating for MoeModel.")
flags.DEFINE_bool(
    "moe_prob_gating", False,
    "Prob gating for MoeModel.")
flags.DEFINE_string(
    "moe_prob_gating_input", "prob",
    "input Prob gating for MoeModel.")
flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    output = slim.fully_connected(
        model_input,
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   is_training,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.
     It also includes the possibility of gating the probabilities

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      is_training: Is this the training phase ?
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
    low_rank_gating = FLAGS.moe_low_rank_gating
    l2_penalty = FLAGS.moe_l2;
    gating_probabilities = FLAGS.moe_prob_gating
    gating_input = FLAGS.moe_prob_gating_input

    input_size = model_input.get_shape().as_list()[1]
    remove_diag = FLAGS.gating_remove_diag

    if low_rank_gating == -1:
        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
    else:
       gate_activations1 = slim.fully_connected(
            model_input,
            low_rank_gating,
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates1")
       gate_activations = slim.fully_connected(
            gate_activations1,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates2")


    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    probabilities = tf.reshape(probabilities_by_class_and_batch,
                                     [-1, vocab_size])

    if gating_probabilities:
        if gating_input == 'prob':
            gating_weights = tf.get_variable("gating_prob_weights",
              [vocab_size, vocab_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(vocab_size)))
            gates = tf.matmul(probabilities, gating_weights)
        else:
            gating_weights = tf.get_variable("gating_prob_weights",
              [input_size, vocab_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(vocab_size)))
 
            gates = tf.matmul(model_input, gating_weights)
        
        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,probabilities)

        gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_prob_bn")

        gates = tf.sigmoid(gates)

        probabilities = tf.multiply(probabilities,gates)


    return {"predictions": probabilities}


# class MoeModel(models.BaseModel):
#   """A softmax over a mixture of logistic models (with L2 regularization)."""
#   def create_model(self,
#                    model_input,
#                    vocab_size,
#                    num_mixtures=None,
#                    l2_penalty=1e-8,
#                    **unused_params):
#     """Creates a Mixture of (Logistic) Experts model.

#      The model consists of a per-class softmax distribution over a
#      configurable number of logistic classifiers. One of the classifiers in the
#      mixture is not trained, and always predicts 0.

#     Args:
#       model_input: 'batch_size' x 'num_features' matrix of input features.
#       vocab_size: The number of classes in the dataset.
#       num_mixtures: The number of mixtures (excluding a dummy 'expert' that
#         always predicts the non-existence of an entity).
#       l2_penalty: How much to penalize the squared magnitudes of parameter
#         values.

#     Returns:
#       A dictionary with a tensor containing the probability predictions of the
#       model in the 'predictions' key. The dimensions of the tensor are
#       batch_size x num_classes.
#     """
#     num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

#     gate_activations = slim.fully_connected(
#         model_input,
#         vocab_size * (num_mixtures + 1),
#         activation_fn=None,
#         biases_initializer=None,
#         weights_regularizer=slim.l2_regularizer(l2_penalty),
#         scope="gates")
#     expert_activations = slim.fully_connected(
#         model_input,
#         vocab_size * num_mixtures,
#         activation_fn=None,
#         weights_regularizer=slim.l2_regularizer(l2_penalty),
#         scope="experts")

#     gating_distribution = tf.nn.softmax(
#         tf.reshape(
#             gate_activations,
#             [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
#     expert_distribution = tf.nn.sigmoid(
#         tf.reshape(expert_activations,
#                    [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

#     final_probabilities_by_class_and_batch = tf.reduce_sum(
#         gating_distribution[:, :num_mixtures] * expert_distribution, 1)
#     final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
#                                      [-1, vocab_size])
#     return {"predictions": final_probabilities}

class ConvNet(models.BaseModel):
  """Custom ConvNet architecture with Residual block with Conv layers """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-3,
                   **unused_params):
    """
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    x = tf.reshape(model_input, [-1,model_input.get_shape().as_list()[1], 1])
    #conv1d
    conv1 = tf.layers.conv1d(x, 32, 7, strides=2, use_bias=True, padding='SAME', name='_conv1') 
    #maxpool1d
    maxpool1 = tf.layers.max_pooling1d(conv1, 4, strides=2,name='maxpool1',padding='VALID')
    maxpool2_flat = tf.layers.flatten(maxpool1)
    #Final FC Layer
    output = slim.fully_connected(
              maxpool2_flat,
              vocab_size,
              activation_fn=tf.nn.sigmoid,
              weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class ResNet(models.BaseModel):
  """Custom ResNet architecture with Residual block with Conv layers """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-3,
                   **unused_params):
    """
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """

    #Residual block
    def resblock(x, filters, kernel_size, name):
        resblock = tf.layers.conv1d(x, filters, kernel_size, strides=1, use_bias=True, padding='SAME', name=name+'_conv1') 
        resblock = tf.layers.batch_normalization(resblock, name=name+'_batch1')
        resblock = tf.nn.relu(resblock, name=name+'_relu')
        resblock = tf.layers.conv1d(resblock, filters, kernel_size, strides=1, use_bias=True, padding='SAME', name=name+'_conv2') 
        resblock = tf.layers.batch_normalization(resblock, name=name+'_batch2')
        return tf.nn.relu(resblock+x)

    x = tf.reshape(model_input, [-1,model_input.get_shape().as_list()[1], 1])
    #conv1d
    conv1 = tf.layers.conv1d(x, 32, 7, strides=2, use_bias=True, padding='SAME', name='_conv1') 
    #maxpool1d
    maxpool1 = tf.layers.max_pooling1d(conv1, 4, strides=2,name='maxpool1',padding='VALID')
    #Number of Residual blocks in the architecture
    #resblock1,2,3
    resblock1=resblock(maxpool1, 32, 3, name='resblock1')
    resblock2=resblock(resblock1, 32, 3, name='resblock2')
    resblock3=resblock(resblock2, 32, 3, name='resblock3')
    maxpool2=tf.layers.max_pooling1d(resblock3, 4, strides=4, name='maxpool2', padding='VALID')
    #print("Maxpool2 shape",maxpool2.get_shape().as_list())
    maxpool2_flat = tf.layers.flatten(maxpool2)
    #Final FC Layer
    output = slim.fully_connected(
              maxpool2_flat,
              vocab_size,
              activation_fn=tf.nn.sigmoid,
              weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class ResNetChanged(models.BaseModel):
  """ResNet architecture with Residual block with Fully Connected layers """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-3,
                   **unused_params):
    """
    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """

    #Residual block
    def resblockblock(x, name):
      #FC+RELU+BATCH_NORM*2+Fc
        #resblock = tf.layers.conv1d(x, filters, kernel_size, strides=1, use_bias=True, padding='SAME', name=name+'_conv1') 
        x = tf.layers.flatten(x)
        output_shape = x.get_shape().as_list()[1]   
        resblock = slim.fully_connected(x,output_shape,activation_fn=tf.nn.relu,weights_regularizer=slim.l2_regularizer(l2_penalty))
        #resblock = tf.nn.relu(resblock, name=name+'_relu1')
        resblock = tf.layers.batch_normalization(resblock, name=name+'_batch1')
        
        # resblock = tf.layers.flatten(resblock)
        # #output_shape = resblock.get_shape().as_list()[1]   
        # resblock = slim.fully_connected(resblock,output_shape,activation_fn=tf.nn.softmax,weights_regularizer=slim.l2_regularizer(l2_penalty))
        # resblock = tf.nn.relu(resblock, name=name+'_relu2')
        # resblock = tf.layers.batch_normalization(resblock, name=name+'_batch2')
        
        resblock = tf.layers.flatten(resblock)
        #output_shape = resblock.get_shape().as_list()[1]   
        resblock = slim.fully_connected(resblock,output_shape,activation_fn=None,weights_regularizer=slim.l2_regularizer(l2_penalty))
        return tf.nn.relu(resblock+x)
    #Number of Residual blocks in the architecture
    
    #x = tf.reshape(model_input, [-1,model_input.get_shape().as_list()[1], 1])
    #resblock1,2,3
    #resblock1=resblockblock(x, vocab_size*4, name='resblock1')

    #resblock = tf.layers.flatten(resblock1)
    #output_shape = resblock.get_shape().as_list()[1]   
    #resblock = slim.fully_connected(resblock,vocab_size*2,activation_fn=tf.nn.softmax,weights_regularizer=slim.l2_regularizer(l2_penalty))    
    #esblock = tf.layers.batch_normalization(resblock, name='_batch11')
        
    # resblock2=resblockblock(resblock,vocab_size*2, name='resblock2')

    # resblock = tf.layers.flatten(resblock2)
    # #output_shape = resblock.get_shape().as_list()[1]   
    # resblock = slim.fully_connected(resblock,vocab_size,activation_fn=tf.nn.softmax,weights_regularizer=slim.l2_regularizer(l2_penalty))    
    # resblock = tf.layers.batch_normalization(resblock, name='_batch12')

    resblock3=resblockblock(model_input, name='resblock3')
    output = slim.fully_connected(resblock3,vocab_size,activation_fn=tf.nn.sigmoid,weights_regularizer=slim.l2_regularizer(l2_penalty))
    
    #resblock = tf.nn.relu(resblock3, name='_relu')
    #output = tf.nn.softmax(resblock, name='softmax')
    return {"predictions": output}
