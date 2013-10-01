#!/usr/bin/python
# Generate testcases.h

import sys
import os


def gen_testcase(fpath):
    # Get name of the test case
    func = os.path.split(fpath)[1].split('.')[0]
    fpath = func + ".cu"
    
    # Get comment of the test case, i.e., the starting comment texts
    # in fpath.
    comment = ""
    try:
        fin = open(fpath, "r")
        line = fin.readline()
        while len(line) > 0 and line.startswith("//"):
            line = line[2:].strip()
            if len(line) > 0:
                if len(comment) > 0:
                    line = " " + line
                comment += line
            line = fin.readline()
        fin.close()
    except IOError as e:
        sys.stderr.write("I/O error(%d): %s\n" % e.errno, e.strerror)
        return -1
    comment = comment.replace("\"", "\\\"")
    
    # Now generate testcases.h
    try:
        fout = open("testcases.h", "w")
        fout.write("#include \"test.h\"\n\n");
        fout.write("// Test cases\n");
        fout.write("int " + func + "();\n\n")
        fout.write("struct test_case testcases[] = {\n")
        fout.write("\t{\n")
        fout.write("\t\t" + func + ",\n")
        fout.write("\t\t\"" + comment + "\"\n")
        fout.write("\t},\n")
        fout.write("};\n")
        fout.close()
    except IOError as e:
        sys.stderr.write("I/O error(%d): %s\n" % e.errno, e.strerror)
        return -1
    
    return 0

def main(args):
    if len(args) < 2:
        sys.stderr.write("USAGE: ./tcgen src_file/obj_file\n")
        sys.exit(1)
    if gen_testcase(args[1]) < 0:
        sys.stderr.write("Failed to generate testcases.h for " + args[1] + "\n")
        return -1
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
