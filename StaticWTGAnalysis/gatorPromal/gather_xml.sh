#! /bin/bash

mkdir -p xml_output

find log_output -regextype posix-egrep -regex '.*\.log' | xargs -I{} grep 'XML file' {} | cut -d':' -f2 | sed -r "s/[[:cntrl:]]\[[0-9]{1,3}m//g" | xargs -I{} cp {} xml_output
