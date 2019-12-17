import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

command = "$NORMA/bin/fr/basic-tokenizer.pl - | $NORMA/bin/fr/generic-normalisation.sh - | $NORMA/bin/fr/specific-normalisation.pl $NORMA/cfg/nlp.cfg -"
command = command.replace("$NORMA", "irisa-text-normalizer")

input_filename = "../../AI_DATA/fr_data.txt"
output_filename = "../../AI_DATA/fr_data_out.txt"

f_in = open(input_filename, "r")

def process_line(line):
    n, line = line
    line = line.replace("\"","\\\"")
    output = subprocess.check_output("echo \"%s\" | %s"%(line, command), shell=True, stderr=subprocess.DEVNULL)
    if n%50==49:
        percent = n/nb_lines
        d = datetime.now()-t0
        print("\r%5.2f%% (time left:%s)"%(percent*100, str(d/percent*(1-percent))), end='')
    return output.decode("utf-8").replace("\n", " ")+"\n"

nb_lines = int(subprocess.check_output("wc -l %s"%input_filename, shell=True).split(b" ")[0])
print("processing %s lines ...."%nb_lines)
t0 = datetime.now()

executor = ThreadPoolExecutor(max_workers=32)
lines_out = executor.map(process_line,enumerate(f_in.readlines()))

f_out= open(output_filename, "wb")
for line in lines_out:
    f_out.write(line.encode("utf-8"))
print("fin")




#
#
#for r,d,files in os.walk("input"):
#    for f in files:
#        print("processing file '%s'."%f)
#        os.system("sh ../PycharmProjects/AI_ENSSAT_s5/irisa-text-normalizer/bin/fr/txt2norm.sh ../PycharmProjects/AI_ENSSAT_s5/irisa-text-normalizer/cfg/nlp.cfg input/%s > output/%s"%(f,f))
#
#print("the end")
