from typing import *
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re

import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt

from course.course_collection import CoursePDFCollection
from course.course_data import CourseDataCollection
from course.utils import cut, pad
from transformer import StureGPT, StureGPTSchedule, masked_accuracy, masked_loss
from transformer.generator import Generator
from transformer.export import Exporter

from course.new_tokens import ret_new_tok_count

from utils import plot_attention_head, plot_attention_weights

import argparse

devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)

courses_folder   = 'a_kursplan/kursinfo-course-plans'
converted_folder = 'a_kursplan/converted-course-plans'
rebuilt_folder   = 'a_kursplan/rebuilt-course-plans'

collection = CoursePDFCollection.from_folder(courses_folder)
collection = collection.filter_by_latest_revision()
collection = collection.filter_by_course_group('dv')
if not os.path.exists(converted_folder):
    collection.convert_from_pdf_to_text(courses_folder, converted_folder)

data_collection = CourseDataCollection.from_folder(converted_folder)
if not os.path.exists(rebuilt_folder):
    data_collection.build_txt_library(rebuilt_folder)

ds, tokenizer = data_collection.build_tensorflow_dataset(
    sequence_length=257,
    BATCH_SIZE=16,
    SHUFFLE_BUFFER_SIZE=100,
    verbose=True,
    force_rebuild_file=True)
    
sturegpt = StureGPT(num_layers=4,
                    model_dims=256,
                    num_heads=4,
                    ff_dims=512,
                    vocab_size=tokenizer.vocab_size + ret_new_tok_count(),      # + nr_new_toks to accompany the additional tokens
                    dropout=0.2)

optimizer = tf.keras.optimizers.Adam(
    # StureGPTSchedule(256, warmup_steps=),
    learning_rate=2e-4,
    beta_1=0.9,
    beta_2=0.99,
    epsilon=1e-9
)

sturegpt.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)

def main(args : Dict[str, Tuple[bool, str]]):
    if args.load_model:
        if args.verbose:
            print(f'Loading model from {"models/sturegpt"}')
        gen = Generator.load_model(tokenizer, sturegpt, 'models/sturegpt', max_length=args.num_tokens)
    else:
        if args.verbose:
            print(f'Rebuilding and training model')
        gen = Generator(tokenizer, sturegpt, max_length=args.num_tokens)
        
    if args.num_epochs > 0:
        gen.transformer.fit(ds, epochs=args.num_epochs)
        gen.save('models/sturegpt')
    else:
        for batch in ds.rebatch(1).take(1):
            gen.transformer(batch[0])
            
    if args.verbose:
        sturegpt.summary()

    context = '''KURSPLAN
dv1278
Revision 2
Fortsättning i php
Continuation in php
Högskolepoäng: 6'''

    if args.verbose:
        print(f'Generating output')
    text, tokens, attention_heads = gen(context)
    print(text)
    print(tokens)

    print(context + '\n' + text.replace('[NL]', '\n'))
    
    plot_attention_weights(context, tokens, tf.squeeze(attention_heads, 0), tokenizer)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', '-e', type=int, default=10, help='How many epochs to train the model on', dest='num_epochs')
    parser.add_argument('--num-tokens', '-n', type=int, default=20, help='How many tokens to generate', dest='num_tokens')
    parser.add_argument('--load-model', '-L', action='store_true', help='Turn on to load a current model', dest='load_model')
    parser.add_argument('--verbose', '-v', action='store_true', help='Turn on verbose', dest='verbose')
    args = parser.parse_args()
    
    main(args)