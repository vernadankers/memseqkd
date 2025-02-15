#!/bin/sh

python hallucinations.py --langpair $1 --srclang $2 --trglang $3 --source $4 --target $5 --models $7 --setup $6 --nathal
python hallucinations.py --langpair $1 --srclang $2 --trglang $3 --source $4 --target $5 --models $7 --setup $6 --oschal --m 10
wait