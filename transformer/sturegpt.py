import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
import numpy

from keras import Sequential, Model
from keras.layers import Layer, Embedding, MultiHeadAttention, Dense, Normalization, Add, Dropout

import numpy as np

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(Layer):
    def __init__(self, vocab_size : int, model_dims : int, *args, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dims = model_dims
        
        # Layers
        self.embedding = Embedding(vocab_size, model_dims, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=model_dims)
        
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, x):
        length = tf.shape(x)[1] # tf.shape(x) -> (batch, length, ...)
        
        # Embed data
        x = self.embedding(x)
        
        # Mix with positional encoding data 
        x *= tf.math.sqrt(tf.cast(self.model_dims, tf.float32))
        x += self.pos_encoding[tf.newaxis, :length, :]
        return x

class AttentionBase(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mha  = MultiHeadAttention(*args, **kwargs)
        self.norm = Normalization()
        self.add  = Add()
        
class AttentionCross(AttentionBase):
    def call(self, x, context):
        att_output, att_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )
        self.last_att_scores = att_scores
        
        x = self.add([x, att_output])
        x = self.norm(x)
        return x, att_scores

class AttentionSelfGlobal(AttentionBase):
    def call(self, x):
        att = self.mha(
            query=x,
            value=x,
            key=x
        )
        x = self.add([x, att])
        x = self.norm(x)
        return x

class AttentionSelfCausal(AttentionBase):
    def call(self, x):
        att = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True
        )
        x = self.add([x, att])
        x = self.norm(x)
        return x
    
class FeedForward(Layer):
    def __init__(self, model_dims : int, ff_dims : int, dropout : float = 0.2):
        super().__init__()
        self.ff = Sequential([
            Dense(ff_dims, activation='relu'),
            Dense(model_dims),
            Dropout(dropout)
        ])
        self.add = Add()
        self.norm = Normalization()
        
    def call(self, x):
        _x = self.ff(x)
        x = self.add([x, _x])
        x = self.norm(x)
        return x

class EncoderLayer(Layer):
    def __init__(self, model_dims : int, num_heads : int, ff_dims : int, dropout : float = 0.2):
        super().__init__()
        self.self_attention = AttentionSelfGlobal(
            num_heads=num_heads,
            key_dim=model_dims,
            dropout=dropout
        )
        self.ff = FeedForward(
            model_dims=model_dims, 
            ff_dims=ff_dims, 
            dropout=dropout
        )
    
    def call(self, x):
        x = self.self_attention(x)
        x = self.ff(x)
        return x

class Encoder(Layer):
    def __init__(self, 
                 num_layers : int, 
                 model_dims : int, 
                 num_heads : int, 
                 ff_dims : int, 
                 vocab_size : int, 
                 dropout : float = 0.2):
        super().__init__()
        
        self.model_dims = model_dims
        self.num_layers = num_layers
        
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, 
            model_dims=model_dims
        )
        
        self.enc_layers = [
            EncoderLayer(model_dims=model_dims,
                         num_heads=num_heads,
                         ff_dims=ff_dims,
                         dropout=dropout)
            for _ in range(num_layers)]
        
        self.dropout = Dropout(dropout)
        
    def call(self, x):
        # Embed
        x = self.pos_embedding(x) # (batch, seq_length, model_dims)
        
        # Dropout
        x = self.dropout(x)
        
        # Encode
        for encoder in self.enc_layers:
            x = encoder(x)
            
        return x # (batch, seq_length, model_dims)

class DecoderLayer(Layer):
    def __init__(self, 
                 model_dims : int, 
                 num_heads : int, 
                 ff_dims : int, 
                 dropout : float = 0.2):
        super().__init__()
        
        self.causal_attention = AttentionSelfCausal(
            num_heads=num_heads,
            key_dim=model_dims,
            dropout=dropout
        )
        self.cross_attention = AttentionCross(
            num_heads=num_heads,
            key_dim=model_dims,
            dropout=dropout
        )
        self.ff = FeedForward(model_dims=model_dims, ff_dims=ff_dims)
        
    def call(self, x, context):
        x = self.causal_attention(x)
        x, attention_scores = self.cross_attention(x=x, context=context)
        
        # save last attention scores
        self.last_att_scores = self.cross_attention.last_att_scores
        
        x = self.ff(x)
        return x, attention_scores
    
class Decoder(Layer):
    def __init__(self, 
                 num_layers : int,
                 model_dims : int,
                 num_heads : int,
                 ff_dims : int,
                 vocab_size : int,
                 dropout : float = 0.2):
        super().__init__()
        self.model_dims = model_dims
        self.num_layers = num_layers
        
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            model_dims=model_dims
        )
        
        self.dropout = Dropout(dropout)
        self.dec_layers = [
            DecoderLayer(model_dims=model_dims,
                         num_heads=num_heads,
                         ff_dims=ff_dims,
                         dropout=dropout)
            for _ in range(num_layers)
        ]
        
        self.ff = FeedForward(model_dims=model_dims, ff_dims=ff_dims, dropout=dropout)
        
        self.last_att_scores = None

    def call(self, x, context):
        x = self.pos_embedding(x)
        
        x = self.dropout(x)
        for dec in self.dec_layers:
            x, attention_scores = dec(x, context)

        self.last_att_scores = self.dec_layers[-1].last_att_scores
        
        x = self.ff(x)
        return x, attention_scores

class StureGPT(Model):
    def __init__(self,
                 num_layers : int,
                 model_dims : int,
                 num_heads  : int,
                 ff_dims    : int,
                 vocab_size : int,
                 dropout    : float = 0.2):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers,
                               model_dims=model_dims,
                               num_heads=num_heads,
                               ff_dims=ff_dims,
                               vocab_size=vocab_size,
                               dropout=dropout)

        self.decoder = Decoder(num_layers=num_layers,
                               model_dims=model_dims,
                               num_heads=num_heads,
                               ff_dims=ff_dims,
                               vocab_size=vocab_size,
                               dropout=dropout)
        
        self.ff = Dense(vocab_size)
    
    # @tf.function(input_signature=[(tf.TensorSpec(shape=(None, None), dtype=tf.int64, name='input_1'), tf.TensorSpec(shape=(None, None), dtype=tf.int64, name='input_2'))])
    def call(self, inputs, return_attention_weights : bool = False):
        # inputs is data of form (context, x)
        ctx, x = inputs
        ctx = self.encoder(ctx)
        
        x, attention_weights = self.decoder(x, ctx)
        
        logits = self.ff(x)
        
        try:
            del logits._keras_mask
        except AttributeError:
            pass
        
        if return_attention_weights:
            return logits, attention_weights
        return logits