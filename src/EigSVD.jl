using LinearAlgebra, SparseArrays, MKLSparse, MKL

function check_mkl()
	return string(BLAS.get_config())
end

# [U, S, V] = eig_svd(A) for m >> n, using eig(A'*A)
function eig_svd(A::AbstractMatrix)
	V = A' * A
	F = eigen(V)
	S = abs.(F.values)	# V may have negative eigenvalues in julia
	V = F.vectors
	V1 = deepcopy(V)	
	n_rows, n_cols = size(V1)
	Threads.@threads for i in 1:n_cols
		S[i] = sqrt(S[i])
		for j in 1:n_rows
			V1[j, i] /= S[i]
		end
	end
	U = A * V1
	return U, S, V
end