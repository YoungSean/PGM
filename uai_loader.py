import numpy as np

from Graph import Graph, RV, F, Domain


def load(f):
    with open(f, 'r') as file:
        file.readline()
        file.readline()

        domains = file.readline().strip().split()

        domain_dict = dict()
        for d in domains:
            if d not in domain_dict:
                domain_dict[d] = Domain(np.arange(int(d)))

        rvs = dict()
        for i, d in enumerate(domains):
            rvs[i] = RV(domain_dict[d])

        num_factor = int(file.readline().strip())

        fs = dict()
        for i in range(num_factor):
            line = file.readline().strip().split()
            fs[i] = F(
                table=None,
                nb=[rvs[int(rv)] for rv in line[1:]]
            )

        file.readline()

        for i in range(num_factor):
            table = list()
            file.readline()
            line = file.readline()
            while line and line != '\n':
                table.append(
                    [float(val) for val in line.strip().split()]
                )
                line = file.readline()
            fs[i].table = np.array(table).reshape([len(rv.domain.values) for rv in fs[i].nb])

    return rvs, fs


if __name__ == '__main__':
    rvs, fs = load('sample_markov.uai')

    for i, rv in rvs.items():
        print(rv.domain.values)

    for i, f in fs.items():
        print(f.table)
