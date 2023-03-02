from typing import *
import tensorflow as tf

class Tokenizer:
    def __init__(self, 
                 keep_line_breaks : bool = True,
                 line_break_token : str = '<NL>',
                 unknown_token : str = '<UNK>'):
        self.vocab = [unknown_token]
        self.unknown_token = unknown_token
        
        self.keep_line_breaks = keep_line_breaks
        self.line_break_token = line_break_token
        
        if self.keep_line_breaks:
            self.add_token(line_break_token)
        
        self.built = False
        
    def __iter__(self):
        for item in self.vocab:
            yield item
        
    def add_token(self, token : str) -> None:
        if not token in self.vocab:
            self.vocab.append(token)
            
    def build_maps(self):
        self._tok2id = {tok:id for id, tok in enumerate(self.vocab)}
        self._id2tok = {id:tok for id, tok in enumerate(self.vocab)}
        self.built = True
    
    def tokenize_token(self, token : str) -> int:
        if token in self._tok2id:
            return self._tok2id[token]
        return self._tok2id[self.unknown_token]
    
    def tokenize(self, s : Union[str, List[str], tf.Tensor]) -> List[int]:
        if not self.built:
            raise ValueError(f'Maps are not built for tokenization. Consider calling build_maps()')
        if isinstance(s, str):
            s = s
        else:
            s = s.numpy().flatten()[0].decode('utf8')
        s = self.rebuild_with_line_breaks(s).split()
        s = tf.ragged.constant([[[self.tokenize_token(token) for token in s]]])
        return s
    
    def detokenize(self, lst : List[int]) -> str:
        if not self.built:
            raise ValueError(f'Maps are not built for detokenization. Consider calling build_maps()')
        
        a = tf.gather(lst, self.vocab, axis=-1)
        return lst
            
    def rebuild_with_line_breaks(self, s : str) -> str:
        if self.keep_line_breaks:
            s = s.replace('\n', f' {self.line_break_token} ')
        return s
        
def tokenizer_from_tf_dataset(ds : tf.data.Dataset, *args, **kwargs) -> Tokenizer:
    tokenizer = Tokenizer(*args, **kwargs)
    total_vocab = set()
    for batch in iter(ds):
        for example in batch:
            example = example.numpy().flatten()
            for item in example:
                item = item.decode('utf8')
                if tokenizer.keep_line_breaks:
                    item = tokenizer.rebuild_with_line_breaks(item)
                item = item.split()
                
                vocab = set(item)
                total_vocab = total_vocab.union(vocab)

    final_vocab = sorted(list(total_vocab))
    for token in final_vocab:
        tokenizer.add_token(token)
    tokenizer.build_maps()
    return tokenizer