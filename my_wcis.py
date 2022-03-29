from uai_loader import load
from Graph import Graph
from VE import VE
import sys
import numpy as np
import os
import time


filepath = "sample_markov.uai"
order_file = "sample_order"
output_file = "sample_cutset"

def run_on_uai_order(filepath, w, order_file, output_file):
    rvs, fs = load(filepath)
    print("Using network {} with {}".format(os.path.basename(filepath), order_file))
    g = Graph(
        rvs={rv for _, rv in rvs.items()},
        fs={f for _, f in fs.items()}
    )
    infer = VE(g, targets={})
    infer.set_order(order_file)
    #print(infer.given_order_names)
    #print(infer.given_order)
    infer.run_use_given_order()
    print(infer.clusters)
    C = infer.find_wCutset(w)
    print("The wCutset is ", C)
    wcutset = sorted(list(C))
    size = len(wcutset)
    with open(output_file, "w") as f:
        f.write(str(size))
        f.write(" ")
        f.write(" ".join([str(i) for i in wcutset]))



run_on_uai_order(filepath, 0, order_file, output_file)