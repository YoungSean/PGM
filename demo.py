from uai_loader import load
from Graph import Graph
from VE import VE
from BP import BP


rvs, fs = load('sample_markov.uai')

# print(rvs)
#
# for i, f in fs.items():
#     print(f.nb)
#     print(f.table)

g = Graph(
    rvs={rv for _, rv in rvs.items()},
    fs={f for _, f in fs.items()}
)

infer = VE(g, targets={})
# infer.run()
# infer.print_fs()
# infer.ve_min_degree()
# infer.print_fs()

infer.ve_min_fill()
infer.print_fs()
# infer.eliminate_rv(rvs[1])
# infer.print_fs()
# infer.eliminate_rv(rvs[2])
# infer.print_fs()

# print(infer.prob(rvs[2]))
