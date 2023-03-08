import tensorflow as tf
from .generator import Generator

class Exporter(tf.Module):
    def __init__(self, generator : Generator):
        self.generator = generator
        self.max_length = 100
        
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, context):
        (result, tokens, attention) = self.generator(context, max_length=self.max_length)
        return result