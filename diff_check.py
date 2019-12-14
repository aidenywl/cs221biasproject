from difflib import SequenceMatcher, Differ
import sys

f1 = sys.argv[1]
f2 = sys.argv[2]

text1 = open(f1).read()
text2 = open(f2).read()
m = SequenceMatcher(None, text1, text2)
print(m.ratio())
