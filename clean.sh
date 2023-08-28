folder=$(date +%Y-%m-%d-%H:%M:%S)
mkdir -p ./trash/$folder
mv nohup.out ./trash/$folder
mv loss_compare.log ./trash/$folder
