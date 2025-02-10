import os
import shutil


bb_dir = "bbox_txt/"
new_bb_dir = "new_bbox_txt/"

if os.path.isdir(new_bb_dir) == False:
    os.makedirs(new_bb_dir)


def is_wrong(txt_path):
    return txt_path.count('_txt') != 0


def png_in_name(txt_path):
    return txt_path.count('_png.') != 0 or txt_path.count('_PNG.')


def rename(txt_path, f_path):
    new_f_path = txt_path.replace('_txt', '_jpg')
    new_f_path = os.path.join(new_bb_dir, new_f_path)
    os.rename(f_path, new_f_path)
    return new_f_path


if __name__ == "__main__":
    for f in os.listdir(bb_dir):
        f_path = os.path.join(bb_dir, f)
        new_f_path = f.replace('_txt', '_jpg')
        print(f_path, new_f_path)
        if os.path.getsize(f_path) != 0 and is_wrong(f):
            new_f_path = os.path.join(new_bb_dir, new_f_path)
            print("The file name has been changed: old file path - " + f_path
                  + " | new file path " + new_f_path)
            shutil.copy(f_path, new_f_path)
            print("The file name has been changed: old file path - " + f_path
                  + " | new file path " + new_f_path)
        elif os.path.getsize(f_path) == 0 and is_wrong(f):
            right_f_path = os.path.join(bb_dir, new_f_path)
            new_f_path = os.path.join(new_bb_dir, new_f_path)
            print("The file name has been changed: old file path - " + right_f_path
                  + " | new file path " + new_f_path)
            shutil.copy(right_f_path, new_f_path)
            print("The file name has been changed: old file path - " + right_f_path
                  + " | new file path " + new_f_path)
        elif png_in_name(f):
            new_f_path = os.path.join(new_bb_dir, f)
            print("The file name has been changed: old file path - " + f_path
                  + " | new file path " + new_f_path)
            shutil.copy(f_path, new_f_path)