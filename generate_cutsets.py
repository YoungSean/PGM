from my_wcis import run_on_uai_order
import os

ws = {1,3,7}
dirpath = "networks"
filenames = os.listdir("networks")
print(filenames)
uai_files = []
order_files = []
pr_files = []

for name in filenames:
    if name.endswith("uai"):
        uai_files.append(name)

    elif name.endswith("order"):
        order_files.append(name)

    else:
        pr_files.append(name)

print("uai files: ", uai_files)
print("order files: ", order_files)

for uai, order in zip(uai_files, order_files):
    uai_path = os.path.join(dirpath, uai)
    order_path = os.path.join(dirpath, order)
    for w in ws:
        out_file = uai + "."+str(w)+"cutset"
        out_file_path = os.path.join("cutsets", out_file)
        run_on_uai_order(uai_path, w, order_path, out_file_path)