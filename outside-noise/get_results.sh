#!/bin/bash

for d in $1/*; do
    r=$(awk 'BEGIN{a=1} $5 < a {a=$5}END{print a, NR}' $d/stat.txt)
    echo $(basename $d) $r
done
