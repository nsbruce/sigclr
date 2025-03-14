from sigclr.metrics import evaluate_similarity_matrix
import numpy as np

def test_similarity_matrix_evaluation():
    perfect_diagonal = np.zeros((8,8))
    for i in range(8):
        perfect_diagonal[i,i] = 1
    
    assert evaluate_similarity_matrix(perfect_diagonal) == 1

    worst_case = np.ones((10,10))
    for i in range(10):
        worst_case[i,i] = 0
    
    assert evaluate_similarity_matrix(worst_case) == -1

    also_bad_result = np.ones((30,30)) * 0.5
    
    assert evaluate_similarity_matrix(also_bad_result) == 0

