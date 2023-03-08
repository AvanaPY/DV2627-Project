

import tensorflow as tf
from .sturegpt import StureGPT
from course.utils import cut, pad
from transformer import masked_accuracy, masked_loss

class Generator():
    def __init__(self, tokenizer, transformer):
        super().__init__()
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.max_length = 10

    def __call__(self, context : str):
        context = self.tokenizer.encode(context)
        context = tf.expand_dims(tf.convert_to_tensor(context, dtype=tf.int64), 0)
        
        start_end = self.tokenizer.encode('')
        start = tf.convert_to_tensor(start_end[0], dtype=tf.int64)[tf.newaxis]
        end   = tf.convert_to_tensor(start_end[1], dtype=tf.int64)[tf.newaxis]
        
        stack = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        stack = stack.write(0, start)

        for i in range(self.max_length):
            pc_stack = tf.transpose(stack.stack())
            preds = self.transformer((context, pc_stack), training=False)
            preds = preds[:, -1, :]
            outs = tf.argmax(preds, axis=-1)
            
            stack = stack.write(i+1, outs)
            
            if outs == end:
                break
        
        tokens = tf.transpose(stack.stack())[0]
        text = self.tokenizer.decode(tokens)
        
        try:
            self.transformer((context, tf.expand_dims(tokens[:-1], 0)), training=False)
            attention_weights = self.transformer.decoder.last_att_scores
        except:
            print(f'WARNING::Attention weights no available!')
            attention_weights = None
            
        return text, tokens, attention_weights
    
    def save(self, path):
        self.transformer.save(path)
        
    def load(self, path):
        self.transformer = tf.keras.models.load_model(
            path,
            custom_objects={
                'masked_loss':masked_loss,
                'masked_accuracy':masked_accuracy
            })