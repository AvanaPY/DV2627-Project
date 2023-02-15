import os
from PyPDF2 import PdfReader
from course.course_collection import Course, CourseCollection
from course.course_section import CourseData

courses_path = 'kursinfo-course-plans'
courses = list(sorted(os.listdir(courses_path)))

collection = CourseCollection.from_list(courses)
collection = collection.filter_by_latest_revision()
collection = collection.filter_by_course_group('dv')

# collection.convert_from_pdf_to_text(courses_path, 'text_courses')

test_course = os.path.join('text_courses', 'dv2627_rev1-00__avancerad_maskininlarning.txt')
course_data = CourseData.from_txt_file_path(test_course)

print(course_data.get_section_by_identifier('3.2'))