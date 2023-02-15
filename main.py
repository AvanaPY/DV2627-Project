import os
import re
from PyPDF2 import PdfReader
from course.course_collection import CoursePDFInfo, CoursePDFCollection
from course.course_data import CourseData

courses_path = 'kursinfo-course-plans'
courses = list(sorted(os.listdir(courses_path)))

collection = CoursePDFCollection.from_list(courses)
collection = collection.filter_by_latest_revision()
collection = collection.filter_by_course_group('dv')

# collection.convert_from_pdf_to_text(courses_path, 'text_courses')

test_course = os.path.join('text_courses', 'dv2627_rev1-00__avancerad_maskininlarning.txt')
test_course = os.path.join('text_courses', 'dv2512_rev4-00__masterarbete_i_datavetenskap.txt')
test_course = os.path.join('text_courses', 'dv1402_rev3-00__unix_och_linux_en_oversikt_och_introduktion.txt')
test_course = os.path.join('text_courses', 'dv1543_rev4-00__skripting_och_andra_sprak.txt')
# test_course = os.path.join('text_courses', 'dv1558_rev2-00__tillampad_programmering_i_java.txt')
course_data = CourseData.from_txt_file_path(test_course)


formatted_data = course_data.redesign_filen_för_fan()
formatted_data = re.sub('([\(\)\[\]\-_:;\!\?,.\\\/\'"”\d•])', r' \1 ', formatted_data)

vocab = sorted(list(set((formatted_data.split()))))
print(f'Vocab size: {len(vocab)}')
print(vocab)