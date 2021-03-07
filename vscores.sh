for dir in save/*/log_validation.txt
do
	echo "$dir"
	head -1 $dir
	echo ""
done

