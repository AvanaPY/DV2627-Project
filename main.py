import os
from course_collection import Course, CourseCollection


courses_path = 'kursinfo-course-plans'
courses = list(sorted(os.listdir(courses_path)))

collection = CourseCollection.from_list(courses)
groups = collection.get_course_groups()

for group in groups:
    group_collection = collection.filter_by_course_group(group)
    group_collection.copy_to_folder(courses_path, os.path.join('course_plans', group))
    # group_collection = group_collection.filter_by_latest_revision()
