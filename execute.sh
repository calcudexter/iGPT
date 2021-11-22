if [ $# -eq 1 ]
then
	python3 src/run.py --size "s" --mdir "models" --ccdir "clusters" --link $1
else
    echo Please Enter Link to the image to be processed by igpt as a command line argument
fi

