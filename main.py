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

ds, tokenizer = data_collection.build_tensorflow_dataset(verbose=True)
    
sturegpt = StureGPT(num_layers=4,
                    model_dims=128,
                    num_heads=4,
                    ff_dims=256,
                    vocab_size=tokenizer.vocab_size+1,      # +1 to accompany the [NL] token
                    dropout=0.2)

optimizer = tf.keras.optimizers.Adam(
    # StureGPTSchedule(256, warmup_steps=),
    learning_rate=1e-4,
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
        gen = Generator.load_model(tokenizer, sturegpt, 'models/sturegpt')
    else:
        if args.verbose:
            print(f'Rebuilding and training model.')
        gen = Generator(tokenizer, sturegpt)
        gen.transformer.fit(ds, epochs=1)
        gen.save('models/sturegpt')

    context = '''KURSPLAN
    dv2627
    Revision 13
    Ett väldigt dåligt program
    Emelie
    Högskolepoäng: -4'''

    if args.verbose:
        print(f'Generating output')
    text, tokens, attention_heads = gen(context)
    print(text)
    print(tokens)

    print(context + '\n' + text.replace('[NL]', '\n'))
    
    plot_attention_weights(context, tokens, tf.squeeze(attention_heads, 0), tokenizer)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', '-L', action='store_true', help='Turn on to load a current model', dest='load_model')
    parser.add_argument('--verbose', '-v', action='store_true', help='Turn on verbose', dest='verbose')
    args = parser.parse_args()
    
    main(args)