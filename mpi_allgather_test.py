import numpy as np
from mpi4py import MPI

def get_all_states(comm, state):
	size = comm.Get_size()
	all_states = np.zeros((size, len(state)), dtype = np.float)
	comm.Allgather([state, MPI.FLOAT], [all_states, MPI.FLOAT])
	return all_states

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
state = np.zeros(6, dtype = np.float) # [x, y, z, row, pitch, yaw]

# put random number for the state of each node
for i in range(0, len(state)):
	state[i] = rank + np.absolute(np.random.normal(loc = 0.0, scale = 0.1, size = 1))

all_states = get_all_states(comm, state)
print("my rank:", rank, ", my state:", state, "\nall states:", all_states)