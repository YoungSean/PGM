from uai_loader import load
from Graph import Graph
from VE import VE
import sys
import numpy as np
import os


def run_on_uai(filepath, order):
    rvs, fs = load(filepath)
    print("Using network {} with {}".format(os.path.basename(filepath), order))
    g = Graph(
        rvs={rv for _, rv in rvs.items()},
        fs={f for _, f in fs.items()}
    )
    infer = VE(g, targets={})
    if order == "mindegree":
        infer.ve_min_degree()
    elif order == "minfill":
        infer.ve_min_fill()
    else:
        print("You need to set order as mindegree or minfill")

    return infer.width, infer.order


def main():
    filename = sys.argv[1]
    ordering = sys.argv[2]
    k = int(sys.argv[3])
    output_filename = sys.argv[4]
    treewidth = sys.maxsize
    width_results = []
    best_order = []
    for i in range(k):
        width, ve_order = run_on_uai(filename, ordering)
        if width < treewidth:
            treewidth = width
            best_order = ve_order
        width_results.append(width)

    width_arr = np.array(width_results)
    mean_width = np.mean(width_arr)
    deviation_width = np.std(width_arr)
    print("************* Experiment Result **************")
    print("Width of the best order on {} = {}".format(os.path.basename(filename), treewidth))
    print("Average and Standard over {} runs = ({:.1f}, {:.2f})".format(k, mean_width, deviation_width))
    print("Best order: ", best_order)
    print("*************************************")

    with open(output_filename, 'w') as f:
        nums = [str(rv) for rv in best_order]
        f.write(" ".join(nums))


if __name__ == '__main__':
    main()
