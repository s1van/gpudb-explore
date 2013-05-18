#!/bin/bash

usage()
{
        echo "Usage: `echo $0| awk -F/ '{print $NF}'`  [-option]"
        echo "[option]:"
        echo "  -s      seconds"
        echo "          seconds waiting before next condition checking "
        echo "  -c      condition "
        echo "                  expression evaluated until 0 "
        echo "  -v      value "
        echo "                  evaluate condition against the value "
        echo
        echo "  e.g     wait4.sh -t 10 -c 'jps|grep RunJar|wc -l' "
        echo
        echo "Copyright by Siyuan Ma  2011-12."
        echo
}

if [ $# -lt 2 ]
then
        usage
        exit
fi

VALUE=0
while getopts "ts:c:v:" OPTION
do
        case $OPTION in
                s)
                        INTV=$OPTARG;
                        ;;
                c)
                        COND=$OPTARG;
                        ;;
                v)
                        VALUE=$OPTARG;
                        ;;
                ?)
                        usage
                        exit
                        ;;
        esac
done


TIMER=
while true; do
    DONE=$(eval $COND)
    if [ $DONE -ne $VALUE ]; then
        sleep $INTV
    else
        break
    fi
done

