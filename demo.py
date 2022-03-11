from uai_loader import load
from Graph import Graph
from VE import VE
import sys
import numpy as np
import os
import time

# rvs, fs = load('sample_markov.uai')
# rvs, fs = load('.\\data\\Grids_11.uai')

# print(rvs)
#
# for i, f in fs.items():
#     print(f.nb)
#     print(f.table)



# infer.run()
# infer.print_fs()
# infer.ve_min_degree()
# infer.print_fs()


def run_on_uai(filepath, order):
    rvs, fs = load(filepath)
    print("Using network {} with {}".format(os.path.basename(filepath), order))
    g = Graph(
        rvs={rv for _, rv in rvs.items()},
        fs={f for _, f in fs.items()}
    )
    infer = VE(g, targets={})
    if order=="mindegree":
        infer.ve_min_degree()
    elif order=="minfill":
        infer.ve_min_fill()
    else:
        print("You need to set order as mindegree or minfill")

    return infer.width, infer.order


def demo_dir(network_dir, order, k=100):
    filenames = os.listdir(network_dir)
    file_paths = [os.path.join(network_dir, f) for f in filenames]

    for f in file_paths:
        demo_file(f, order, k)

def demo_file(f, order, k):
    output_filename = os.path.basename(f) + "." + order + "_order"+ "_" + str(k) + "_runs"
    # print(output_filename)
    start_time = time.time()
    experiment(f, order, k, output_filename)
    average_time = (time.time() - start_time) / k
    print("---Average time %s seconds ---" % average_time)


def experiment(f, order, kRuns, output_order):

    # run_on_uai('.\\data\\Grids_11.uai', 'mindegree')
    filename = f #sys.argv[1]
    ordering = order # sys.argv[2]
    k = kRuns # int(sys.argv[3])
    output_filename = output_order #sys.argv[4]
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
    # demo_file("sample_markov.uai", "minfill", 3)
    # demo_dir(os.path.join(".", "data"), "mindegree", 3)
    # demo_file(os.path.join(".", "data", "ObjectDetection_11.uai"), "mindegree", 3)
    demo_file(os.path.join(".", "data", "Segmentation_11.uai"), "minfill", 5)