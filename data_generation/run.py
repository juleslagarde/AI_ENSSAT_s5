import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

command = "$NORMA/bin/fr/basic-tokenizer.pl - | $NORMA/bin/fr/generic-normalisation.sh - | $NORMA/bin/fr/specific-normalisation.pl $NORMA/cfg/nlp.cfg -"
command = command.replace("$NORMA", "irisa-text-normalizer")

#input_filename = "./sample2_fr_data.txt"
#output_filename = "./sample2_fr_data_out.txt"
input_filename = "../../AI_DATA/fr_data.txt"
output_filename = "../../AI_DATA/fr_data_out.txt"
batch_size = 1000

f_in = open(input_filename, "r")
start_line=0

def process_line(n,line):
    line = line.replace("'","''")
    output = subprocess.check_output("echo '%s' | %s"%(line, command), shell=True, stderr=subprocess.DEVNULL)
    if n%50==49:
        percent = (n-start_line)/nb_lines
        d = datetime.now()-t0
        print("\r%5.2f%% (time left :%s)"%(n/nb_lines*100, str(d/percent*(1-percent))), end='')
    return output.decode("utf-8").replace("\n", " ")+"\n"


def calc_lines(filename):
    """
    crash if file doesn't exist
    """
    return int(subprocess.check_output("wc -l %s"%filename, shell=True).split(b" ")[0])

def write_lines(f, lines):
    for line in lines:
        f.write(line.encode("utf-8"))


nb_lines = calc_lines(input_filename)
print("processing %s lines ...."%nb_lines)

with open(output_filename, "ab") as f_out:  # create output file before 'wc -l' command
    nb_lines_out = calc_lines(output_filename)

    if nb_lines_out > 0:
        print("%s lines already processed"%nb_lines_out)
        if nb_lines_out%batch_size != 0:
            print("ERROR: %s lines missmatch batch_size=%s"%(nb_lines_out,batch_size))
            sys.exit(42)
        print("starting at line %s from input file"%(nb_lines_out+1))
        start_line = nb_lines_out

    t0 = datetime.now()
    lines = f_in.readlines()
    for i in range(start_line, len(lines), batch_size):
        executor = ThreadPoolExecutor(max_workers=32)
        lines_in = lines[i:i+batch_size]
        lines_out = list(executor.map(process_line,range(i,i+batch_size),lines_in))
        if len(lines_out) != len(lines_in):
            s = "(%s) batch %s has outputed %s lines but was given %s lines (%s)"%(input_filename,i/batch_size,len(lines_out),len(lines_in))
            print(s)
            with open("error_lines.txt", "ab") as f:
                f.write(s.encode("utf-8"))
                write_lines(f, lines_out)
            sys.exit(42)
        write_lines(f_out, lines_out)
        f_out.flush()
        if calc_lines(output_filename) != i+len(lines_in):
            s="(%s) write batch %s failed"%(input_filename,i/batch_size)
            print(s)
            with open("error_lines.txt", "ab") as f:
                f.write(s.encode("utf-8"))
                write_lines(f, lines_out)
            sys.exit(42)

        print("\n%s lines done"%(i+len(lines_out)))

print("fin")




#
#
#for r,d,files in os.walk("input"):
#    for f in files:
#        print("processing file '%s'."%f)
#        os.system("sh ../PycharmProjects/AI_ENSSAT_s5/irisa-text-normalizer/bin/fr/txt2norm.sh ../PycharmProjects/AI_ENSSAT_s5/irisa-text-normalizer/cfg/nlp.cfg input/%s > output/%s"%(f,f))
#
#print("the end")
