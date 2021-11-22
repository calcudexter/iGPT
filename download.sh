#!/bin/bash

if [ $# -eq 1 ]
then
    curl $1 > sg.jpeg   
else
    echo Please Enter Link to the image to be processed by igpt as a command line argument
fi