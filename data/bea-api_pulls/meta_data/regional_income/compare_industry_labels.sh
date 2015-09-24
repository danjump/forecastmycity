#!/bin/bash
grep "For table" full_table_info.txt | sed 's/For table //g' | uniq  > unique_names.txt
grep "For table" full_table_info.txt | sed 's/For table "\([^"]*\).*/\"\1\"/g' | uniq | xargs -I {} grep -c \"{}\" full_table_info.txt > counts.txt

paste counts.txt unique_names.txt > table_linecounts.txt

rm counts.txt
rm unique_names.txt
