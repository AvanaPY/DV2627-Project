from typing import *
from collections import namedtuple
import os
import re

processedCourse = namedtuple('ProcessedCourse', 'course_group course_code revision course_name')

def process_course_file_name(course : str) -> processedCourse:
    """
        Takes in a file name and processes it. This requires the file name to be of a specific format, namely \"<course code>_<revision_id>-<some number>__<course name>.pdf\"
        
    
    """
    code_rev, name = course.split('__', 1)
    code, revision = code_rev.split('_', 1)
    
    try:
        r = re.compile('([a-zA-Z]+)(\d+)')
        code_grp, num = r.match(code).group(1), r.match(code).group(2)
        return processedCourse(code_grp, num, revision, name)
    except AttributeError:
        print(f'Cannot match \"{code}\"')

def filter_by_course_group(course_group : str) -> Callable:
    """
        Returns a wrapper that returns whether or not an input string starts with a given sequence.
    """
    def wrapper(x : processedCourse) -> bool:
        return x[0].startswith(course_group)
    return wrapper

def filter_by_latest_revision(courses : List[processedCourse]) -> List[processedCourse]:
    """
        Filters a list of course names by the latest revision.
    """
    course_groups = {}
    for group, code, revision, name in courses:
        group_code = group + code
        if group_code in course_groups:
            course_groups[group_code] += [(revision, name)]
        else:
            course_groups[group_code] = [(revision, name)]
    
    cs = []
    for group_code, revision_list in course_groups.items():
        rev, name = revision_list[-1]
        
        r = re.compile('([a-zA-Z]+)(\d+)')
        code_grp, num = r.match(group_code).group(1), r.match(group_code).group(2)
        
        cs.append(processedCourse(code_grp, num, rev, name))
    return cs

def filter_courses_by(courses : List[str], course_group : Optional[str] = None, revision : Optional[str] = None):
    """
        Generic function that takes a list of course file names and filters them accordingly.
    """
    grps = []
    processed_coures = list(map(process_course_file_name, courses))
    
    if course_group:
        processed_coures = list(filter(filter_by_course_group(course_group), processed_coures))

    if revision:
        if revision == 'latest':
            processed_coures = filter_by_latest_revision(processed_coures)

    return processed_coures

courses_path = 'kursinfo-course-plans'
courses = list(sorted(os.listdir(courses_path)))

grps = filter_courses_by(courses, 'dv', 'latest')
print(f'Number of filtered courses: {len(grps)}')
for processed_course in grps[:20]:
    print(f'\t{processed_course}')