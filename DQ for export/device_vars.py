train_f_path = None ## add the file path to train.py
train_multi_path = None # add the file location of train_multi.py
python_path = None # add the file location of your python here
has_gpu = False # keep it as false, since I'm using CPU training, have some switching funciton somewhere
#   but havent' updated it for a while

i_start = -54
i_end = -2

a = ["seeded-by-week/", "data/NQ1-1m-DATA-BY-WEEK/"] # just folder names
b = ["seeded-TUE-MON/", "data/NQ1-1m-DATA-TUE-MON/"] # just folder names
c = ["seeded-WED-TUE/", "data/NQ1-1m-DATA-WED-TUE/"] # just folder names
d = ["seeded-THU-WED/", "data/NQ1-1m-DATA-THU-WED/"] # just folder names
e = ["seeded-FRI-THU/", "data/NQ1-1m-DATA-FRI-THU/"] # just folder names

sel = [a, b, c, d, e]

save_fold = [a[0] for a in sel]
data_fold = [a[1] for a in sel]

threads = 3
use_cmd = True

train_1_day = False