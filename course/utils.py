from typing import *
from dataclasses import dataclass
import re

@dataclass
class CourseFileInfo:
    course_code : str
    revision : str
    course_name : str
    file_type : str

def extract_data_from_file_name(file_name : str) -> Tuple[str, str, str, str]:
    expr = re.compile('([a-zA-Z]+\d+)_rev(\d+)\-\d+__(.+)\.(.+)')
    matches = expr.match(file_name)
    course_code = matches.group(1)
    revision = matches.group(2)
    course_name = matches.group(3)
    file_type = matches.group(4)
    return course_code, revision, course_name, file_type

def cut(sequence : str, sequence_length : int) -> List[int]:
    if len(sequence) > sequence_length:
        return sequence[:sequence_length]
    return sequence

def pad(sequence : List[int], sequence_length : int) -> List[int]:
    if len(sequence) < sequence_length:
        return sequence + ([0] * (sequence_length - len(sequence)))
    return sequence