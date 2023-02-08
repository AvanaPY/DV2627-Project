from __future__ import annotations
from typing import *

import os
import shutil
import re
import tqdm

class Course:
    def __init__(self, group : str, code : str, revision : str, name : str):
        self._group = group
        self._code = code
        self._revision = revision
        self._name = name
    
    def __str__(self):
        return f'Course[Group={self._group}, Code={self._code}, Revision={self._revision}, Name={self._name}]'
    
    def __repr__(self):
        return str(self)
    
    @property
    def group(self):
        return self._group
    
    @property
    def code(self):
        return self._code
    
    @property
    def revision(self):
        return self._revision
    
    @property
    def name(self):
        return self._name
    
    def build_course_id(self):
        return f'{self.group}{self.code}'
    
    def rebuild_path(self):
        return f'{self.group}{self.code}_{self.revision}__{self.name}.pdf'
    
class CourseCollection:
    MAX_PRINT_SIZE = 50
    def __init__(self):
        self._collection = []
    
    def __str__(self) -> str:
        s = f'CourseCollection of {self.length} courses:\n'
        for i, item in zip(range(CourseCollection.MAX_PRINT_SIZE), self._collection):
            s += f'\t{i:5d}: {item}\n'
        if self.length > CourseCollection.MAX_PRINT_SIZE:
            s += f'\t...'
        return s
    
    def __getitem__(self, x : int) -> Course:
        return self._collection[x]
    
    @property
    def length(self) -> int:
        return len(self._collection)
    
    def add_from_path(self, path : str) -> None:
        code_rev, name = path.split('__', 1)
        code, revision = code_rev.split('_', 1)
        name, ext = os.path.splitext(name)
        
        try:
            r = re.compile('([a-zA-Z]+)(\d+)')
            code_grp, num = r.match(code).group(1), r.match(code).group(2)
        except AttributeError:
            print(f'Cannot match \"{code}\"')
            
        self._collection.append(Course(code_grp, num, revision, name))
    
    def filter_by_course_group(self, course_group : str) -> CourseCollection:
        collection = filter(lambda x : x.group == course_group, self._collection)
        collection = list(map(lambda x : x.rebuild_path(), collection))
        return CourseCollection.from_list(collection) 
    
    def filter_by_latest_revision(self) -> CourseCollection:
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
        return CourseCollection.from_list(collection_list)
        
    def group_by_course_id(self) -> Dict[str, Course]:
        collection = {}
        for item in self._collection:
            course_id = item.build_course_id()
            if course_id in collection:
                collection[course_id] += [item]
            else:
                collection[course_id] = [item]
        return collection
    
    def copy_to_folder(self, old_folder : str, new_folder_path : str):
        os.makedirs(new_folder_path, exist_ok=True)
        
        print(f'Copying {self.length} files...')
        for item in tqdm.tqdm(self._collection):
            file_name = item.rebuild_path()
            
            old_path = os.path.join(old_folder, file_name)
            new_path = os.path.join(new_folder_path, file_name)
            shutil.copyfile(old_path, new_path)
    
    def get_course_groups(self):
        groups = sorted(list(set([a.group for a in self._collection])))
        return groups
    
    @staticmethod
    def from_list(lst : List[str]) -> CourseCollection:
        collection = CourseCollection()
        for item in lst:
            collection.add_from_path(item)
            
        return collection