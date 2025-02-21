import os

bb_dir = "new_bbox_txt/"


if __name__ == "__main__":
    c = 0
    for f in os.listdir(bb_dir):
        if os.path.getsize(bb_dir + f) != 0:
            with open(bb_dir + f, "r+") as f:
                elems = f.readline().split()
                while elems:
                    if elems and len(elems) != 5:
                        f.truncate(0)
                        c += 1
                    elems = f.readline().split()
    print(c)
