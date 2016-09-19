for dir in ./[0-9]*; do
    if [ -d "$dir" ]; then
        cd "$dir"
        echo "$dir"
        make -f Makefile.osx
        # make clean 
        cd ..
    fi
done

