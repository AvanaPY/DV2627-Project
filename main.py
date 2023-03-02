import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re

import tensorflow as tf
import keras
from keras import layers

from course.course_collection import CoursePDFInfo, CoursePDFCollection
from course.course_data import CourseData, CourseDataCollection

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
    
ds, tokenizer = data_collection.build_tensorflow_dataset(verbose=True)

for ctx, x in ds.take(1):
    print(f'CTX: {ctx}')
    print(f'TOK: {tokenizer.tokenize(ctx)}')
    print(f'''DET: {tf.strings.reduce_join(
        tokenizer.detokenize(tokenizer.tokenize(ctx).merge_dims(-2,-1)),
        separator=" "
    ).numpy().decode("utf8").replace("   ", " ")}''')