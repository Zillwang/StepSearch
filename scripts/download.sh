save_path=/mnt/GeneralModel/zhengxuhui/data/stepsearch

python scripts/download.py --save_path $save_path

cat $save_path/part_* > e5_Flat.index
