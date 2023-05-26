using LinearAlgebra, SparseArrays, MKLSparse, MKL

"""
Check if MKL is installed correctly.
"""
function check_mkl()
	return string(BLAS.get_config())
end

"""
[U, S, V] = eig_svd(A) for m >= n, using eigen(A'*A)
"""
function eig_svd(A::AbstractMatrix)
	m, n = size(A)
	@assert m >= n
	V = A' * A
	F = eigen(LinearAlgebra.Hermitian(V))
	S = F.values
	V = F.vectors
	V1 = deepcopy(V)	
	n_rows, n_cols = size(V1)
	Threads.@threads for i in 1:n_cols
		S[i] = sqrt(S[i])
		V1[:, i] ./= S[i]
	end
	U = A * V1
	return U, S, V
end