class State:
    def __init__(self, idx):
        self.id        = idx
        self.neighbors = []

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return str(self.id)

class Graph:

    def __init__(self, filename):
        self.states = {}

        with open(filename, 'r') as f:
            for line in f.readlines():
                line_split = line.split()
                id_from = int(line_split[0])
                id_to   = int(line_split[1])

                # Is state with id_from already existing?
                if id_from not in self.states:
                    self.states[id_from] = State(id_from)
                
                # Is state with id_to already existing?
                if id_to not in self.states:
                    self.states[id_to] = State(id_to)
                
                # Store edge between the two neighbors
                self.states[id_from].neighbors.append(id_to)


class BFS:

    def solve(self, graph, start_id, end_id):
        # BFS: You have to solve it on your own!
        visited = {} # Dictionary mapping child->parent edges in state expansion
        queue   = [start_id] # BFS queue
        
        sequence = []  # Output sequence of states

        # while len(queue) > 0:
            # TODO: fill BFS in here
        
        return sequence

filename  = 'ukol2_graph.txt'

graph = Graph(filename)
print(graph)
bfs   = BFS()

start_idx = 2
end_idx   = 5
sequence = bfs.solve(graph, start_idx, end_idx)

if len(sequence) == 0:
    print('Solution not found.')
else:
    # print('s0:', s0)
    for s in sequence:
        print(s)
