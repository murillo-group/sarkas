
# This shell script should be run to purge the docs folder of old website files such as
# .rst and .html files.

# Instructions: From the command line enter "$ bash gen_website.sh" and a new index.html file
# and .rst files for all current Sarkas and utilities .py files will be generated with 
# auto-documentation.

#!/bin/bash 

# PATH to Sarkas
SARKAS_PATH=~/sarkas

# Clear old files from directory
rm -r ./doctrees/*
rm -r ./html/*
rm -r ./rst_files/auto_gen_rst/sarkas_rst/*
# rm -r ./rst_files/auto_gen_rst/utilities_rst/*

# Generate .rst files
sphinx-apidoc -o ./rst_files/auto_gen_rst/sarkas_rst/ $SARKAS_PATH/.
# sphinx-apidoc -o ./rst_files/auto_gen_rst/utilities_rst/ $SARKAS_PATH/utilities/

# Make a copy of the .rst files to be read by make command
cp -r ./rst_files/usr_gen_rst/*.rst .

# Make the website
make html

# Make a copy of the .html files in the docs folder for GitHub hosting
cp -r ./html/* .

# Clean .html files out of .rst folders
rm -r ./rst_files/usr_gen_rst/*.html