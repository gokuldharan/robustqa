for dir in save/*/log*validation.txt
do
	echo "$dir"
	head -1 $dir
	echo ""
done

