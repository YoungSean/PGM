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

infer = VE(g, targets={rvs[2]})
infer.run()
# infer.eliminate_rv(rvs[0])
# infer.print_fs()

print(infer.prob(rvs[2]))
