from __future__ import annotations
from typing import *
from dataclasses import dataclass
import re
import os
from tqdm import tqdm

import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset

from tokenizer.tokenizer import Tokenizer, tokenizer_from_tf_dataset
from course.utils import extract_data_from_file_name

@dataclass(frozen=True)
class CourseSection:
    identifier : str
    title : str
    content : str

class CourseData:
    def __init__(self, 
                 swe_course_name : str,
                 eng_course_name : str,
                 ects : str,
                 sections : List[CourseSection], 
                 filename : Optional[str] = None):
        self.swe_course_name = swe_course_name
        self.eng_course_name = eng_course_name
        self.ects = ects
        self.sections : List[CourseSection] = sections
        self.filename = filename
        course_id, revision, course_name, file_type = extract_data_from_file_name(filename)
        self.course_id = course_id
        self.revision = revision
        self.course_name = course_name
        self.file_type = file_type

    def __str__(self) -> str:
        return '\n'.join([
            f'CourseData for {self.swe_course_name} ({self.eng_course_name})',
            f'  ECTS: {self.ects}',
            f'  Sections:',
            self.get_printable_sections(),
            f'  Filename: {self.filename}'
        ])

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
            s.append(f'{section.identifier.ljust(5, " ")} : {section.title.ljust(50, ".")} {section.content[:50]}...')
        return '\n'.join(s)
    
    def redesign_filen_för_fan(self) -> str:
        
        file_content : List[str] = []
        file_content.append('KURSPLAN')
        file_content.append(self.course_id)
        file_content.append(f'Revision {self.revision}')
        file_content.append(self.swe_course_name)
        file_content.append(self.eng_course_name)
        
        file_content.append(f'Högskolepoäng: {self.ects}')
        
        for section in self.sections:
            file_content.append(f'{section.identifier} {section.title}')
            file_content.append(section.content)
    
        file_content = '\n'.join(file_content)
        file_content = re.sub(' ([:.,])', r'\1', file_content)
        file_content = re.sub(' +', ' ', file_content)
       
        return file_content
        
    def get_as_datapoint(self):
        
        context : List[str] = []
        
        context.append('KURSPLAN')
        context.append(self.course_id)
        context.append(f'Revision {self.revision}')
        context.append(self.swe_course_name)
        context.append(self.eng_course_name)
        context.append(f'Högskolepoäng: {self.ects}')
        
        context = '\n'.join(context)
        context = re.sub(' ([:.,])', r'\1', context)
        context = re.sub(' +', ' ', context)
        
        file_content : List[str] = []
        for section in self.sections:
            file_content.append(f'{section.identifier} {section.title}')
            file_content.append(section.content)
    
        file_content = '\n'.join(file_content)
        file_content = re.sub(' ([:.,])', r'\1', file_content)
        file_content = re.sub(' +', ' ', file_content)
        
        return context, file_content
    
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
        regexp = re.compile('^(\d{1,2}\.\d{0,2})')
        
        with open(file_path, 'r') as f:
            data = f.read()
            data = data.replace('\xa0', '')   # Filter out unwanted characters, apparently \xa0 appears a lot
            data = data.split('\n')
            
            before, after = CourseData.split_at_section_one(data)
            
            # Find course name and title
            for i, line in enumerate(before):
                if line.startswith('KURSPLAN'):
                    swe_course_name = before[i + 1].strip()
                    eng_course_name = before[i + 2].strip()
                    break
                
            full_before = ''.join(before).replace(' ', '')
            expr = re.compile('(\d+,*\d*) *[poäng|högskolepoäng]')
            matches = expr.findall(full_before)
            try:
                ects = matches[0]
            except Exception as e:
                print(f'Error: {e} on file \"{file_path}\". Could not find ECTS.')
                print(f'Full section:')
                print(before)
                print(full_before)
                print(f'Matches for expr {expr}: {matches}')
                exit(0)
                
            # Build sections
            i : int = 0
            while i < len(after):
                line = after[i]
                matches = regexp.match(line)
                if not matches:
                    print(line)
                    print(f'ERROR: No matches for {regexp=} found with {i=} on {file_path=}')
                    exit(0)
                    
                identifier = matches.groups()[0]
                if len(line) > len(matches.groups()[0]):
                    title = line[len(matches.groups()[0])+1:]
                else:
                    try:
                        i += 1
                        title = after[i]
                    except Exception as e:
                        print(f'Error when building title:', len(after), identifier, i, after)
                        exit(0)
                        
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
                    
                content = '\n'.join(content)
                sections.append(CourseSection(
                    identifier=identifier,
                    title=title,
                    content=content
                ))
                
        directory, filename = os.path.split(file_path)
        return CourseData(
            swe_course_name=swe_course_name,
            eng_course_name=eng_course_name,
            ects=ects,
            sections=sections, 
            filename=filename)
        
class CourseDataCollection:
    def __init__(self, collection : List[CourseData]):
        self.collection = collection
        self.vocab_size = 4000
        self.reserved_tokens = [
            '[START]', 
            '[END]', 
            '[UNK]', 
            '[PAD]',
            '[NL]'
        ]
    
    def __str__(self) -> str:
        items = self.items
        return f'CourseDataCollection[{items=}]'
    
    def __getitem__(self, x : Any) -> CourseData:
        return self.collection[x]
    
    @property
    def items(self) -> int:
        return len(self.collection)
    
    def build_txt_library(self, new_folder : str) -> None:
        if not os.path.exists(new_folder):
            os.makedirs(new_folder, exist_ok=True)
        
        print(f'Building text library...')
        for course in tqdm(self.collection, ncols=100):
            data = course.redesign_filen_för_fan()
            filename, ext = os.path.splitext(course.filename)
            with open(os.path.join(new_folder, filename + '.txt'), 'w+') as f:
                f.write(data)
    
    def __get_tokenizer_from_dataset(self, ds : tf.data.Dataset, verbose:bool=False) -> Any:
        
        vocab_path = 'tmp/vocab.txt'
        bert_tokenizer_params = dict(
            keep_whitespace=True)
        if not os.path.exists(vocab_path):
            if verbose:
                print(f'Building vocabulary. This may take a few minutes.')
            os.makedirs(os.path.split(vocab_path)[0], exist_ok=True)
            vocab_args = dict(    
                vocab_size=self.vocab_size,
                reserved_tokens=self.reserved_tokens,
                bert_tokenizer_params=bert_tokenizer_params,
                learn_params={}
            )
            vocab = bert_vocab_from_dataset.bert_vocab_from_dataset(
                ds.batch(100).prefetch(2),
                **vocab_args
            )
            with open(vocab_path, 'w+') as f:
                for token in vocab:
                    print(token, file=f)
            
            if verbose:
                print(f'Vocabulary built: \"{vocab_path}\"')
        
        tokenizer = tf_text.BertTokenizer(vocab_path, **bert_tokenizer_params)
        return tokenizer
    
    def build_tensorflow_dataset(self, verbose:bool=False) -> Tuple[tf.data.Dataset, Tokenizer]:
        dataset_path = 'tmp/dataset.tfds'
        
        # First build a basic dataset and make a tokenizer using tensorflow text
        data = [[item.get_as_datapoint()] for item in self.collection]
        ds = tf.data.Dataset.from_tensor_slices(data)
        
        
        # Build or load the tokenizer
        tokenizer = self.__get_tokenizer_from_dataset(ds, verbose=verbose)
        
        SEQUENCE_LENGTH = 128
        BATCH_SIZE = 1
        BUFFER_SIZE = 1_000
        
        def prepare_batch(data):
            ctx = data[:,:,0]
            # ctx = tokenizer.tokenize(ctx)
            # ctx = ctx.merge_dims(-3, -1)
            # ctx = ctx.to_tensor()
            
            x = data[:,:,1]
            # x = tokenizer.tokenize(x)
            # x = x.merge_dims(-3, -1)
            # x = x.to_tensor()
            return ctx, x
        
        def batch(ds):
            return (
                ds
                # .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .map(prepare_batch, tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
                )
        ds = batch(ds)
        return ds, tokenizer
        
    @staticmethod
    def from_folder(folder_path : str) -> CourseDataCollection:
        courses = os.listdir(folder_path)
        collection : List[CourseData] = []
        for course in courses:
            name, ext = os.path.splitext(course)
            if ext.lower() == '.txt':
                # If it's a txt file, just append it as a txt file
                collection.append(CourseData.from_txt_file_path(os.path.join(folder_path, course)))
            else:
                raise ValueError(f'Non-supported file format found on {course=}')
        return CourseDataCollection(collection=collection)