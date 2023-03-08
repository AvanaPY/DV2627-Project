import os
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

devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)

courses_folder   = 'kursinfo-course-plans'
converted_folder = 'converted-course-plans'
rebuilt_folder   = 'rebuilt-course-plans'

collection = CoursePDFCollection.from_folder(courses_folder)
collection = collection.filter_by_latest_revision()
collection = collection.filter_by_course_group('dv')
if not os.path.exists(converted_folder):
    collection.convert_from_pdf_to_text(courses_folder, converted_folder)

data_collection = CourseDataCollection.from_folder(converted_folder)
if not os.path.exists(rebuilt_folder):
    data_collection.build_txt_library(rebuilt_folder)
    
    
def plot_attention_head(in_tokens, translated_tokens, attention):
    # The model didn't generate `<START>` in the output. Skip it.
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label for label in in_tokens]
    ax.set_xticklabels(
        labels, rotation=90)

    labels = [label for label in translated_tokens]
    ax.set_yticklabels(labels)
    
def plot_attention_weights(sentence, translated_tokens, attention_heads, max_heads : int = 2):
    in_tokens = tokenizer.encode(sentence)
    in_tokens = tokenizer.convert_ids_to_tokens(in_tokens)
    
    translated_tokens = [label for label in translated_tokens.numpy()]
    translated_tokens = tokenizer.convert_ids_to_tokens(translated_tokens)
    
    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads[:max_heads]):
        ax = fig.add_subplot(1, 2, h+1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h+1}')

    # plt.tight_layout()
    plt.show()    

ds, tokenizer = data_collection.build_tensorflow_dataset(verbose=True)

tf_tokenizer = data_collection.get_tf_tokenizer()

sturegpt = StureGPT(num_layers=4,
                    model_dims=128,
                    num_heads=4,
                    ff_dims=256,
                    vocab_size=tokenizer.vocab_size+1,      # +1 to accompany the [NL] token
                    dropout=0.2)

optimizer = tf.keras.optimizers.Adam(
    # StureGPTSchedule(256, warmup_steps=),
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.99,
    epsilon=1e-9
)

sturegpt.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)
for data in ds.take(1):
    sturegpt(data[0])


gen = Generator(tokenizer, sturegpt)
gen.save('models/sturegpt')
gen.load('models/sturegpt')

context = '''KURSPLAN
dv2627
Revision 13
Ett väldigt dåligt program
Emelie
Högskolepoäng: -4'''

text, tokens, attention_heads = gen(context)
print(text)

# text, tokens, att = gen(context, max_length=20)
# print(text)
# print(tokens)


# att = tf.squeeze(att, 0)
# attention = att[0]
# plot_attention_weights(context, tokens, att)