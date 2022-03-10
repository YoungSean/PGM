from uai_loader import load
from Graph import Graph
from VE import VE
from BP import BP


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
    print("Using network", filepath)
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



def main():

    run_on_uai('.\\data\\Grids_11.uai', 'mindegree')

if __name__ == '__main__':
    main()