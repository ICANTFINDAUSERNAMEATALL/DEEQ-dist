def delete_files_in_directory(directory_path):
    import os
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    # os.rmdir(directory_path)

def add_dir(path):
    import os
    os.mkdir(path)

a2 = "save_imgs/seeded-by-week-run/"
b2 = "save_imgs/seeded-fri-thu-run/"
c2 = "save_imgs/seeded-thu-wed-run/"
d2 = "save_imgs/seeded-tue-mon-run/"
e2 = "save_imgs/seeded-wed-tue-run/"
all_folds = [a2, b2, c2, d2, e2]

a1 = "models/seeded-by-week"
b1 = "models/seeded-FRI-THU"
c1 = "models/seeded-THU-WED"
d1 = "models/seeded-TUE-MON"
e1 = "models/seeded-WED-TUE"

af2 = [a1, b1, c1, d1, e1]

# for i in all_folds:
#     add_dir(i)

if input("ARE YOU SURE YOU WANT TO DELETE ALL FILES? [Y/N]: ") == "Y":
    for i in all_folds:
        delete_files_in_directory(i)

if input("ARE YOU SURE YOU WANT TO DELETE ALL MODELS? [Y/N]: ") == "Y":
    for i in af2:
        delete_files_in_directory(i)
