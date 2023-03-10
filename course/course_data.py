from __future__ import annotations
from typing import *
from dataclasses import dataclass
import re
import os
from tqdm import tqdm

import tensorflow as tf
from course.utils import extract_data_from_file_name
from course.utils import cut, pad
from transformers import BertTokenizerFast, TFBertTokenizer

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
        file_content = re.sub('\n', '[NL]', file_content)
        
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
    
    def __get_tokenizer(self, verbose:bool=False) -> BertTokenizerFast:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        tokenizer.add_tokens(["[NL]"], special_tokens=True)
        return tokenizer
    
    def get_tf_tokenizer(self) -> TFBertTokenizer:
        tokenizer = TFBertTokenizer.from_tokenizer(self.__get_tokenizer(False))
        return tokenizer
    
    def build_tensorflow_dataset(self, verbose:bool=False) -> Tuple[tf.data.Dataset, BertTokenizerFast]:
        def prepare_data(data : Tuple[int, int]) -> Tuple[str, str]:
            ctx = data[0]
            ctx = cut(ctx, sequence_length)
            ctx = pad(ctx, sequence_length)
            
            x = data[1]
            x = cut(x, sequence_length)
            x = pad(x, sequence_length)
            return ctx, x
        
        def work(data : List[Tuple[str, str]]) -> List[Tuple[int, int]]:
            d = []
            
            seq_len = sequence_length - 1
            STEP_SIZE = seq_len // 4
            
            for datapoint in tqdm(data):
                context, x = datapoint
                context = tokenizer.encode(context)
                context = context[:seq_len]
                
                x_end = seq_len - STEP_SIZE
                x = tokenizer.encode(x)
                while x_end < len(x):
                    x_end += STEP_SIZE
                    _x = x[x_end-seq_len:x_end]
                    d.append((context, _x))

            return list(map(prepare_data, d))
        
        sequence_length = 256 + 1
        
        # Build and load the tokenizer
        tokenizer = self.__get_tokenizer(verbose=verbose)
        
        # First build a basic dataset and make a tokenizer using tensorflow text
        dataset_path = 'tmp/dataset.tfds'
        
        if os.path.exists(dataset_path):
            ds = tf.data.Dataset.load(dataset_path)
        else:
            data = [item.get_as_datapoint() for item in self.collection]        
            data = work(data)
            
            ds = tf.data.Dataset.from_tensor_slices(data)
            
            BATCH_SIZE = 32
            BUFFER_SIZE = 1_000 
            
            def prepare_batch(data : tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
                ctx = data[0][:-1] # Turn from SEQUENCE_LENGTH to SEQUENCE_LENGTH - 1
                
                xy = data[1]
                x  = xy[:-1]
                y  = xy[1:]
                
                ctx = tf.cast(ctx, dtype=tf.int64)
                x   = tf.cast(x, dtype=tf.int64)
                y   = tf.cast(y, dtype=tf.int64)
                return (ctx, x), y
            
            def batch(ds : tf.data.Dataset) -> Tuple[tf.data.Dataset, BertTokenizerFast]:
                return (
                    ds
                    .map(prepare_batch, tf.data.AUTOTUNE)
                    .shuffle(BUFFER_SIZE)
                    .batch(BATCH_SIZE)
                    .cache()
                    .prefetch(tf.data.AUTOTUNE)
                    )
            ds = batch(ds)
            
            if verbose:
                print(f'Saving dataset to {dataset_path}')
            ds.save(dataset_path)
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