#!/usr/bin/python

# import numpy
# import scipy
# import matplotlib
import os, sys

" Pretty Parsing"
# Pretty Parsing

__author__ = "Xianglan Piao"
__email__ = "xianglan0502@gmail.com"
__version__ = "1.0.0"

def main(argv):
    import optparse
    class MyParser(optparse.OptionParser):
        def format_epilog(self, formatter):
            return self.epilog

    parser = MyParser( 
                     usage = "%prog [options] <command line> [<arg1>...]",
                     version = __version__,
                     epilog =\
"""
Example:
    ./jaws.py gcc hello.c
    ./jaws.py gcc hello.c -l5
    ./jaws.py gcc hello.c -l5 -t one@email
    ./jaws.py gcc hello.c -l5 -t one@email,two@email --timeout=3600
"""
    )

    parser.add_option("-l", "--loop",
                      dest="loop",
                      type="int",
                      default=1,
                      help="repeat the execution. (default: 1)"
    )
    parser.add_option("--sleep",
                      dest="sleep",
                      type="int",
                      default=5,
                      help="set sleep seconds between executions. (default: 5)"
    )
    parser.add_option("--timeout",
                      dest="timeout",
                      type="int",
                      default=0,
                      help="set timeout seconds."
    )
    parser.add_option("-f", "--file",
                      dest="file",
                      type="string",
                      default="jaws.log",
                      help="set log file. (default: \"jaws.log\")"
    )
    parser.add_option("-t", "--to",
                      dest="to",
                      type="string",
                      help="set email address separated by comma(,)"
    )

    (options, args) = parser.parse_args()
    
    if len(args) == 0:
        parser.print_help()
    else:
        text = run(args, options.loop, options.sleep, options.timeout, options.file)    
        if options.to is not None:
            subject = " ".join(args)
            mailto(options.to, subject, text)

def run(cmd, loop=1, sleep=5, timeout=0, f="jaws.log", timer=False, stdout=True):
    import subprocess, time
    from datetime import datetime
    assert isinstance(loop, int)
    assert isinstance(sleep, int)
    assert isinstance(timeout, int)
    assert isinstance(f, str)
    assert isinstance(cmd, list)
    assert isinstance(timer, bool)
    assert isinstance(stdout, bool)

    cmd = map(os.path.expanduser, cmd)
    
    if stdout: 
        print " ".join(cmd)
    
    result = "## %s\n## %s\n" % (datetime.now(), " ".join(cmd))
    fp = open(f, "a+")
    fp.write("%s" % result)
    
    for i in range(0, loop):
        if timeout > 0:
            start = datetime.now()

        if timer is True:
            startT = datetime.now()

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None)
        while p.poll() is None:
            l = p.stdout.readline()
            if stdout: sys.stdout.write(l)
            result = result + l
            fp.write(l)
            if timeout > 0:
                now = datetime.now()
                if (now-start).seconds > timeout:
                    p.kill()
                    print "\033[05;31mTimeout!!!\033[0m"

        l = p.stdout.read()
        if stdout: sys.stdout.write(l)
        result = result + l
        fp.write(l)

        if timer is True:
            endT = datetime.now()
            dT = (endT-startT).total_seconds()
            result = result + str(dT) + " sec"
            fp.write(str(dT) + " sec")
            print "%f sec" % dT

        if loop > 1:
            time.sleep(sleep)

    fp.write("\n")
    fp.close()
    return result

def findall(string, pattern):
    import re
    assert isinstance(string, str)
    try:
        ret = re.findall(pattern, string)
        ret = map(try_parse, ret)
    except:
        print "Cannot find the pattern in the given string"
        ret = []
    return ret

def replace(string, patternFrom, patternTo):
    import re
    assert isinstance(string, str)
    try:
        ret = re.sub(patternFrom, patternTo, string)
    except:
        ret = ""
    return ret

def try_parse(v):
    try:
        ret = float(v)
    except ValueError:
        try:
            ret = complex(v)
        except ValueError:
            ret = v
    return ret

def printCSV(array2d, indicator):
    ret = ""
    for array1d in array2d:
        ret += indicator.join(str(x) for x in array1d)
        ret += "\n"
    return ret

def mailto(TO, subject, text):
    import re
    assert isinstance(TO, str)
    assert isinstance(subject, str)
    assert isinstance(text, str)
    SENDMAIL = "/usr/sbin/sendmail"
    FROM = os.environ['USER'] + "@" + os.uname()[1]
    text = re.sub(r'\033\[\d\d;\d\dm', '', text)
    text = re.sub(r'\033\[0m', '', text)
    message = """\
From: %s
To: %s
Subject: %s
%s
    """ % (FROM, TO, subject, text)
    sys.stdout.write("\nsending email to %s ... " % TO)
    sys.stdout.flush()
    mail = os.popen("%s -t -i" % SENDMAIL, "w")
    mail.write(message)
    status = mail.close()
    if status:
        print "\033[05;31mfail\033[0m", status
    print "\033[05;32msuccess\033[0m"


if __name__ == "__main__":
    main(sys.argv[1:])
