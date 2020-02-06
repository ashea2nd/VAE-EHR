import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from numba import cuda
import math
import pickle
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--startIdx", help="Int for starting patient index", type=int)
parser.add_argument("--endIdx", help="Int for ending patient index (exclusive)", type=int)
args = parser.parse_args()

startIdx = args.startIdx
endIdx = args.endIdx
print("startIdx, endIdx: ({},{})".format(startIdx, endIdx))

patient_icd_path="/data1/andrew/meng/mixehr/data/Mimic/andrew_outputs/PATIENT_ICD_BINARY.csv"
patient_icd_df = pd.read_csv(patient_icd_path, sep=' ')

icd_code_idxs = patient_icd_df.drop("SUBJECT_ID", axis=1).columns
patient_idxs = patient_icd_df.index

patient_data = patient_icd_df.drop('SUBJECT_ID', axis=1).as_matrix().astype(np.float)

@cuda.jit
def euc_sim_matrix(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        sqdist = 0
        for i in range(A.shape[1]):
            sqdist += (A[row, i] - B[i, col])**2
        euc_sim = 1 / (1 + math.sqrt(sqdist)) #Caps similarity at 1 and avoids division by 0
        C[row][col] = euc_sim

def build_sim_matrix():
    global startIdx, endIdx
    assert startIdx != None and endIdx != None, "Tried to build similarity matrix but command line options --startIdx and --endIdx not specified."
    sim_matrix=np.empty(shape=(0, patient_data.shape[0]))

    cuda.select_device(1)
    print("(Start, End): ({}, {})".format(startIdx, endIdx))
    endIdx = min(patient_data.shape[0], endIdx)
    A = np.ascontiguousarray(patient_data[startIdx : endIdx])
    B = patient_data.T
    print(A.shape, B.shape)
    print(type(A[0]), type(B[0]))

    A_mem = cuda.to_device(A)
    B_mem = cuda.to_device(B)
    C_mem = cuda.device_array((A.shape[0], B.shape[1]), dtype=np.float)

    threadsperblock=(16, 16)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
    blockspergrid=(blockspergrid_x, blockspergrid_y)

    euc_sim_matrix[blockspergrid, threadsperblock](A_mem, B_mem, C_mem)
    C = C_mem.copy_to_host()
    print(C.shape)
    print(C)
    pickle.dump(C, open("euc_sim_matrix_patients_{}_{}.p".format(startIdx, endIdx), "wb"))
    cuda.close()

def merge_matrix_chunks():
    chunk1_filename = "euc_sim_matrix_patients_0_10000.p"
    chunk2_filename = "euc_sim_matrix_patients_10000_20000.p"
    chunk3_filename = "euc_sim_matrix_patients_20000_30000.p"
    chunk4_filename = "euc_sim_matrix_patients_30000_40000.p"
    chunk5_filename = "euc_sim_matrix_patients_40000_46520.p"

    sim_matrix = np.zeros((patient_data.shape[0], patient_data.shape[0]))
    sim_matrix[0:10000] = pickle.load(open(chunk1_filename, "rb")) 
    sim_matrix[10000:20000] = pickle.load(open(chunk2_filename, "rb")) 
    sim_matrix[20000:30000] = pickle.load(open(chunk3_filename, "rb")) 
    sim_matrix[30000:40000] = pickle.load(open(chunk4_filename, "rb")) 
    sim_matrix[40000:] = pickle.load(open(chunk5_filename, "rb")) 

    print(sim_matrix.shape)
    print(sim_matrix)

    pickle.dump(sim_matrix, open("euc_sim_matrix.p", "wb"), protocol=4)

if __name__ == "__main__":
    merge_matrix_chunks()



