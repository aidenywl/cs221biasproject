from difflib import SequenceMatcher, Differ
text1 = open("test1.txt").read()
text2 = open("test2.txt").read()
m = SequenceMatcher(None, text1, text2)
print(m.ratio())
