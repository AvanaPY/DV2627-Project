import os
import re
from PyPDF2 import PdfReader
from course.course_collection import CoursePDFInfo, CoursePDFCollection
from course.course_data import CourseData, CourseDataCollection

courses_folder   = 'kursinfo-course-plans'
converted_folder = 'converted-course-plans'
rebuilt_folder   = 'rebuilt-course-plans'

if not os.path.exists(converted_folder):
    collection = CoursePDFCollection.from_folder(courses_folder)
    collection = collection.filter_by_latest_revision()
    collection = collection.filter_by_course_group('dv')
    collection.convert_from_pdf_to_text(courses_folder, converted_folder)

if not os.path.exists(rebuilt_folder) or True:
    data_collection = CourseDataCollection.from_folder(converted_folder)
    data_collection.build_txt_library(rebuilt_folder)