""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols
import warnings

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  text = text.strip()
  clean_text = _clean_text(text, cleaner_names)
  symbols = ""
  for symbol in clean_text:
    if symbol in _symbol_to_id:
      symbol_id = _symbol_to_id[symbol]
      sequence += [symbol_id]
      symbols += symbol
  return sequence


def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  try:
    cleaned_text = cleaned_text.strip()
    sequence = []
    for symbol in cleaned_text:
      if symbol in _symbol_to_id:
        sequence.append(_symbol_to_id[symbol])
      else:
        pass
        # warnings.warn(f"Symbol {symbol} from sequence {cleaned_text} does not exist in the dictionary. Skipping symbol {symbol}")
    # sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  except KeyError as e:
    print(e, file=open("error.log", "w"))
    exit()
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
