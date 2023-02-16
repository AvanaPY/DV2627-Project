from __future__ import annotations
from typing import *

import os
import shutil
import re
import tqdm
from PyPDF2 import PdfReader

class CoursePDFInfo:
    """
        A Class that represents a course file name in a different format.
    """
    def __init__(self, group : str, code : str, revision : str, name : str):
        self._group = group
        self._code = code
        self._revision = revision
        self._name = name
    
    def __str__(self) -> str:
        return f'Course[Group={self._group}, Code={self._code}, Revision={self._revision}, Name={self._name}]'
    
    def __repr__(self) -> str:
        return str(self)
    
    @property
    def group(self) -> str:
        return self._group
    
    @property
    def code(self) -> str:
        return self._code
    
    @property
    def revision(self) -> str:
        return self._revision
    
    @property
    def name(self) -> str:
        return self._name
    
    def build_course_id(self) -> str: 
        """
            Builds the course ID
            
            @Returns :: str
        """
        return f'{self.group}{self.code}'
    
    def rebuild_path(self) -> str:
        """
            Rebuilds the file name.
            
            @Returns :: str
        """
        return f'{self.group}{self.code}_{self.revision}__{self.name}.pdf'
    
class CoursePDFCollection:
    """
        A collection of PDF course syllabus files for the purpose of filtering and grouping courses, as well as converting them to .txt files.
    """
    MAX_PRINT_SIZE = 50
    def __init__(self):
        self._collection : List[CoursePDFInfo] = []
    
    def __str__(self) -> str:
        s = f'CourseCollection of {self.length} courses:\n'
        for i, item in zip(range(CoursePDFCollection.MAX_PRINT_SIZE), self._collection):
            s += f'\t{i:5d}: {item}\n'
        if self.length > CoursePDFCollection.MAX_PRINT_SIZE:
            s += f'\t...'
        return s
    
    def __getitem__(self, x : int) -> CoursePDFInfo:
        return self._collection[x]
    
    @property
    def length(self) -> int:
        return len(self._collection)
    
    def add_from_path(self, path : str) -> None:
        """
            Adds a course from a filepath to the collection.
            
            @param :: path : str - The path to the course syllabus file

            @Returns :: None
        """
        code_rev, name = path.split('__', 1)
        code, revision = code_rev.split('_', 1)
        name, ext = os.path.splitext(name)
        
        try:
            r = re.compile('([a-zA-Z]+)(\d+)')
            code_grp, num = r.match(code).group(1), r.match(code).group(2)
        except AttributeError:
            print(f'Cannot match \"{code}\"')
            
        self._collection.append(CoursePDFInfo(code_grp, num, revision, name))
    
    def filter_by_course_group(self, course_group : str) -> CoursePDFCollection:
        """
            Creates a new filtered CourseCollection based on a course group given
            
            @param :: course_group : str - The course group to filter the collection by
            
            @Returns :: CourseCollection
        """
        collection = filter(lambda x : x.group == course_group, self._collection)
        collection = list(map(lambda x : x.rebuild_path(), collection))
        return CoursePDFCollection.from_list(collection) 
    
    def filter_by_latest_revision(self) -> CoursePDFCollection:
        """
            Creates a new filtered CourseCollection by the latest revisions
            
            @Returns :: CourseCollection
        """
        collections = {}
        for item in self._collection:
            course_id = item.build_course_id()
            if course_id in collections:
                collections[course_id] += [item]
            else:
                collections[course_id] = [item]
        
        collection_list = [
            collections[key][-1] for key in collections.keys()
        ]
        collection_list = list(map(lambda x : x.rebuild_path(), collection_list))
        return CoursePDFCollection.from_list(collection_list)
        
    def group_by_course_id(self) -> Dict[str, CoursePDFInfo]:
        """
            Creates a dictionary that maps course ids to a list of courses containing all revisions in the current collection.
            
            @Returns :: Dict[str, Course]
        """
        collection = {}
        for item in self._collection:
            course_id = item.build_course_id()
            if course_id in collection:
                collection[course_id] += [item]
            else:
                collection[course_id] = [item]
        return collection
    
    def copy_to_folder(self, old_folder_path : str, new_folder_path : str) -> None:
        """
            Copies all files contained in the collection.
            
            @param :: old_folder : str - The source folder where all files in the collection currently exists.
            @param :: new_folder : str - The new folder to which to copy all the files contained in the collection to.

            @Returns :: None
        """
        os.makedirs(new_folder_path, exist_ok=True)
        
        print(f'Copying {self.length} files from {old_folder_path} to {new_folder_path}')
        for item in tqdm.tqdm(self._collection, ncols=100):
            file_name = item.rebuild_path()
            
            old_path = os.path.join(old_folder_path, file_name)
            new_path = os.path.join(new_folder_path, file_name)
            shutil.copyfile(old_path, new_path)
    
    def get_course_groups(self) -> List[str]:
        """
            Creates a list of course groups that are contained in the current collection
            
            @Returns :: List[str]
        """
        groups = sorted(list(set([a.group for a in self._collection])))
        return groups
    
    def convert_from_pdf_to_text(self, original_folder : str, new_folder : str) -> None:
        if not os.path.exists(new_folder):
            os.makedirs(new_folder, exist_ok=True)
            
        print(f'Converting {self.length} PDF files to TXT...')
        for course in tqdm.tqdm(self._collection, ncols=100):
            path = os.path.join(original_folder, course.rebuild_path())
            new_path = os.path.join(new_folder, course.rebuild_path().replace('pdf', 'txt'))
            
            reader = PdfReader(path)
            page_data = [page.extract_text(0) for page in reader.pages]
            data = '\n'.join(page_data)
            with open(new_path, 'w+') as f:
                f.write(data)

    @staticmethod
    def from_folder(folder_path : str) -> CoursePDFCollection:
        courses = list(sorted(os.listdir(folder_path)))
        return CoursePDFCollection.from_list(courses)

    @staticmethod
    def from_list(lst : List[str]) -> CoursePDFCollection:
        """
            Creates a CourseCollection instance from a list of paths to course syllabi.
            
            @param :: lst : List[str] - A list of string instances where each string represents a path to a course syllabus
            
            @Returns :: CourseCollection
        """
        collection = CoursePDFCollection()
        for item in lst:
            collection.add_from_path(item)
            
        return collection