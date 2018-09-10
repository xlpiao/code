for dir in ./[0-9]*; do
    if [ -d "$dir" ]; then
        cd "$dir"
        echo "$dir"
        make -f Makefile.osx
        # make -f Makefile.osx clean
        cd ..
    fi
done

