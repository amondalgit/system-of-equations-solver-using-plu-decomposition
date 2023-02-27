import numpy as np
import helper as hp
import numpy.linalg as LA
import scipy.linalg as SLA
import pandas as pd

from time import perf_counter

# ----i/p: interger (degree); o/p: Identity matrix of that degree----#
def identity(n):
    return np.eye(n)

# ----dot product of two array with asssuming they have same length----#
def dot_product(arr1, arr2):
    n = arr1.shape[0]
    return sum([el1*el2 for el1,el2 in zip(arr1, arr2)])

# cross product of two matrices assuming they are matrrix multiplicable----#
def cross_product(A,B):
    shape1, shape2 = A.shape, B.shape
    if A.ndim == 1:
        if B.ndim == 1:         # both 1-D matrices
            C = np.array(dot_product(A, B))
        else:                   # A 1-D, B row ratrix
            n = shape2[1]
            C = np.zeros((n,))
            for i in range(shape2[1]):
                C[i] = dot_product(A, B[:,i])
    else:
        if B.ndim == 1:         # A rectangular, B column matrix
            n = shape1[0]
            C = np.zeros((n,))
            for i in range(n):
                C[i] = dot_product(A[i,:],B)
        else:                   # both rectangular matrices
            n, m = shape1[0], shape2[1]
            C = np.zeros((n,m))
            for row in range(n):
                for col in range(m):
                    C[row,col] = dot_product(A[row,:], B[:,col])
    return C

# ----i/p: integer n, o/p: random array of length n----#
def get_array(n):
    return np.random.uniform(-100,100,(n,))

# ----i/p: integer n, o/p: random square matrix of degree n----#
def get_matrix(n, random_state = 55):
    np.random.seed(random_state)
    return np.random.uniform(-100,100,(n, n))

# ----swaps any equal length arrays of two rows in a matrix----#
# i/p: operable matrix A, indeces of rows as two intergers, indeces of columns as two integers where portions are to be swapped
# o/p: matrix A after the swap operation

def swap_row(A, diagonal_row, pivot_row, from_col = 0, to_col = -1):
    if to_col == -1: 
        to_col = len(A)
    A[[diagonal_row, pivot_row], from_col:to_col] = A[[pivot_row, diagonal_row], from_col:to_col]
    return A

# ----i/p: array of length n, o/p: index of the element which has max absolute value----#
def absolute_max_element_index(arr):
    arr = list(map(abs, arr))
    arr_len, max_el_index  = len(arr), 0
    max_el = abs(arr[max_el_index])

    for index in range(1,arr_len):
        if arr[index] > max_el:
            max_el_index = index
            max_el = abs(arr[max_el_index]) 
    return max_el_index


# i/p: suqare Upper triangular matrix U with degree n, array b of length n
# o/p: solution array C retained by backward substitution
# where, UC = b

def backward_substitution(mat, arr):
    (n,m) = mat.shape               # n X n square matrix
    C = np.zeros((m,))              # initialization of C
    
    for i in range(n-1,-1,-1):
        substracting_factor = 0 if (i == n-1) else dot_product(mat[i, i+1:], C[i+1:])
        C[i] = (arr[i] - substracting_factor) / mat[i, i]
    return C


# i/p: suqare Unit Lower triangular matrix L with degree n, array b of length n
# o/p: solution array C retained by forward substitution
# where, LC = b

def forward_substitution(mat, arr):
    (n,m) = mat.shape               # n X n square matrix
    C = np.zeros((m,))              # initialization of C

    for i in range(n):
        substracting_factor = 0 if (i == 0) else dot_product(mat[i, :i], C[:i])
        C[i] = (arr[i] - substracting_factor) / mat[i, i]
    return C


#------------i/p: matrix A; o/p: P, L, U using patrial pivoting and Time Taken for the task-----------#
# PA = LU
# P : Permutation matrix
# L : Lower unit triangular matrix
# U : Upper triangular matrix

def LU_partial_pivoting(A):
    degree = A.shape[0]
    L = hp.identity(degree)
    P = hp.identity(degree)
    U = A.copy()
    
    pivot_index_array = []

    start = perf_counter()
    for row in range(degree-1):
        pivot_search_array = U[row:,row]

        if sum(abs(pivot_search_array)) == 0:       # checks if the matrix is sigular
            print(f'the marix is singular and the rank is at least {len(pivot_index_array)}')
            return None
        
        pivot_index = row + hp.absolute_max_element_index(pivot_search_array)
        pivot_index_array.append(pivot_index)       # track pivot index(>=i) in column i for first n-1 columns
        
        if pivot_index != row:
            U = hp.swap_row(U, row, pivot_index, from_col = row)
            if row > 0: 
                L = hp.swap_row(L, row, pivot_index, to_col = row)
            P = hp.swap_row(P, row, pivot_index)

        for row_below in range(row+1, degree):
            pivot_ratio = L[row_below, row] = U[row_below,row] / U[row, row]
            U[row_below,row:] = U[row_below,row:] - (pivot_ratio * U[row,row:])
    
    decomp_time = perf_counter() - start            # total time for decomposition

    return P, L, U, decomp_time


#------------UDF built from scratch for solving system of equations using LU decomp and Partial Pivoting-----------#
# i/p : A (coeff. matrix), b (constant array)
# o/p : x (solution)
# where, PAx = LUx = LC = Pb

def my_system_of_equations_solver(A, b):
    decomp = LU_partial_pivoting(A)
    if decomp == None:
        return None
    
    P, L, U, decomp_time = decomp
    Pb = hp.cross_product(P, b)

    start = perf_counter()
    C = hp.forward_substitution(L, Pb)          # solves for C, where, LC = Pb
    x = hp.backward_substitution(U, C)          # solves for x, where, Ux = C
    solution_time = perf_counter() - start      # time for processing the solution after getting P, L, U

    ax_minus_b_norm = LA.norm(hp.cross_product(A, x) - b)                       # Norm of Ax-b
    pa_minus_lu_norm = LA.norm(hp.cross_product(P, A) - hp.cross_product(L, U)) # Norm of PA-LU

    return {'solution':x, 'axb_norm': ax_minus_b_norm, 'palu_norm':pa_minus_lu_norm, 
            'time_solve':solution_time, 'time_decomp':decomp_time}


#------------UDF for solving system of equations as Scipy as Base-----------#
def scipy_system_of_equations_solver(A, b):
    try:
        # -------------solve using scipy.linalg's lu method-------------
        start = perf_counter()
        P, L, U = SLA.lu(A)
        P = LA.inv(P)               # P(permutation matrix) from scipy.lu is inverse of permutaion matrix P of PAx = LUx = Pb
        decomp_time_lu = perf_counter() - start

        Pb = P@b    
        
        start = perf_counter()
        C = SLA.solve_triangular(L, Pb, lower=True, unit_diagonal=True) # solves for C, where, LC = Pb
        x_lu = SLA.solve_triangular(U, C)                               # solves for x, where, Ux = C
        solution_time_lu = perf_counter() - start                       # time for processing the solution after getting P, L, U

        # -------------solve using scipy.linalg's lu_factor method-------------
        start = perf_counter()
        parameters = (lu, piv) = SLA.lu_factor(A)                       # PLU decomposition using scipy.linalg's lu_factor method
        
        section_ge = perf_counter()
        x_lu_factor = SLA.lu_solve(parameters, b)                       # solve for x using scipy.linalg's lu_factor method

        solution_time_lu_factor = perf_counter() - section_ge           # time taken for PLU decomposition
        decomp_time_lu_factor = section_ge - start                      # time taken for # solve for x

        # calculation of norms
        ax_minus_b_norm_lu  = LA.norm(A@x_lu - b)                       # Norm of Ax-b using scipy.linalg's lu method
        ax_minus_b_norm_lu_factor = LA.norm(A@x_lu_factor - b)          # Norm of Ax-b using scipy.linalg's lu_factor method
        pa_minus_lu_norm_lu = LA.norm(P@A - L@U)                        # Norm of PA-LU using scipy.linalg's lu method
        
        # Norm of PA-LU using scipy.linalg's lu_factor method could not be easily calculated because the Permutation matrix is in
        # LEPACK's permutation array form

        return {'solution':x_lu, 'palu_norm_lu':pa_minus_lu_norm_lu, 'axb_norm_lu': ax_minus_b_norm_lu, 
                'axb_norm_lu_factor':ax_minus_b_norm_lu_factor, 
                'time_solve_lu':solution_time_lu, 'time_decomp_lu':decomp_time_lu, 
                'time_solve_lu_factor':solution_time_lu_factor, 'time_decomp_lu_factor':decomp_time_lu_factor}

    except SLA.LinAlgError:
        print('The marix is singular. Cannot solve for an unique solution')
        return None