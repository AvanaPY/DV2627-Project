from __future__ import annotations
from typing import *
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class CourseSection:
    identifier : str
    title : str
    content : str

class CourseData:
    def __init__(self, sections : List[CourseSection]):
        self.sections : List[CourseSection] = sections

    def get_section_by_identifier(self, identifier : str) -> CourseSection:
        section = list(filter(lambda x : x.identifier == identifier, self.sections))
        if len(section) > 0:
            return section[0]
        return None

    def get_section_by_title(self, title : str) -> CourseSection:
        section = list(filter(lambda x : x.title == title, self.sections))
        if len(section) > 0:
            return section[0]
        return None
    
    def get_printable_sections(self) -> str:
        s = []
        for section in self.sections:
            s.append(f'{section.identifier.rjust(5, " ")} : {section.title.ljust(50, ".")} {section.content[:50]}...')
        return '\n'.join(s)
    
    @staticmethod
    def split_at_section_one(texts : List[str]) -> Tuple[List[str], List[str]]:
        before : List[str] = []
        after : List[str] = []
        found_line : bool = False
        for line in texts:
            if line.startswith('1.'):
                found_line = True
            
            if not found_line:
                before.append(line)
            else:
                after.append(line)
                
        return (before, after)
            

    @staticmethod
    def from_txt_file_path(file_path : str) -> CourseData:
        sections : List[CourseSection] = []
        regexp = re.compile('^(\d+\.\d*)')
        with open(file_path, 'r') as f:
            data = f.read()
            data = data.split('\n')
            
            # Build sections
            before, after = CourseData.split_at_section_one(data)
            i : int = 0
            while i < len(after):
                line = after[i]
                matches = regexp.match(line)
                if not matches:
                    print(line)
                    print(f'ERROR {i}')
                    exit(0)
                    
                identifier = matches.groups()[0]
                if len(line) > len(matches.groups()[0]):
                    title = line[len(matches.groups()[0])+1:]
                else:
                    i += 1
                    title = after[i]
                content = []
                
                # This builds the content of the section
                i += 1
                matches = None
                while not matches and i < len(after):
                    line = after[i]
                    matches = regexp.match(line)
                    if matches:
                        break
                    
                    content.append(line)
                    i += 1
                content = ' '.join(content)
                sections.append(CourseSection(
                    identifier=identifier,
                    title=title,
                    content=content
                ))
                
            
        return CourseData(sections)