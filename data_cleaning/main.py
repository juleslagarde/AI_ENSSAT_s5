import os
import sys
from typing import List

from html2text import HTML2Text
from bs4 import *

MAIN_DIV_ID = "bodyContent"
FILE_BLACKLIST = ["Utilisateur~", "Image~", "Discussion_Utilisateur~", "Utilisateur~", "Discuter~"]


def extractFile(filename: str, h: HTML2Text) -> List[str]:
	f = open(filename, "r")
	try:
		content = f.read()
		if len(content) < 1000:
			return []
		soup = BeautifulSoup(content, features="html.parser")
		f.close()
		result = soup.find_all("div", id=MAIN_DIV_ID)
		if len(result) != 1:
			if len(result) == 0:
				print("Div '%s' not found in file '%s'" % (MAIN_DIV_ID, filename))
			elif len(result) > 1:
				print("Found %s div with id='%s' which one is it ? " % (len(result), MAIN_DIV_ID))
			else:
				assert False
			return []
		div = result[0]
		s = h.handle(str(div))
		s = s.replace("[modifier]", "").replace("\\-", "-")
		return list(filter(lambda x: len(x) > 80, s.split("\n")))
	except UnicodeDecodeError:
		print("File '%s' can't be decoded with UTF-8" % filename)
	return []


def writeFile(filename, texts):
	f = open(filename, "w")
	f.write("\n\n".join(texts))
	f.close()


def main(argv):
	if len(argv) != 3:
		print("Usage: python %s <inputDirectory> <outputDirectory>" % argv[0])
		sys.exit(1)
	inputDir = argv[1]
	if not os.path.isdir(inputDir):
		print("Input directory %s not found" % inputDir)
		sys.exit(1)
	files = []
	for root, d, f in os.walk(inputDir):
		for file in f:
			# ok = True
			# for fbl in FILE_BLACKLIST:
			# 	if file.startswith(fbl):
			# 		ok = False
			# 		break
			# if ok:
			files.append(os.path.relpath(os.path.join(root, file), inputDir))
	lenFiles = len(files)
	print("Input directory found ! (%s files will be processed)" % (lenFiles))
	outputDir = argv[2]
	if not os.path.isdir(outputDir):
		print("Input directory %s not found" % outputDir)
		sys.exit(1)
	h = HTML2Text()
	h.ignore_emphasis = True
	h.ignore_tables = True
	h.ignore_images = True
	h.ignore_links = True
	f = open(os.path.join(outputDir, "input_data2.txt"), "w")
	for i, file in zip(range(lenFiles), files):
		texts = extractFile(os.path.join(inputDir, file), h)
		# writeFile(os.path.join(outputDir, file).replace(".html", ".txt"), texts)
		if len(texts) > 0:
			f.write("\n\n" + ("\n\n".join(texts)))
		if i % 1000 == 0:
			print("Processed %s out of %s (%5.2f%%)" % (i + 1, lenFiles, (i + 1) * 100 / lenFiles))
	f.close()


if __name__ == "__main__":
	main(sys.argv)
