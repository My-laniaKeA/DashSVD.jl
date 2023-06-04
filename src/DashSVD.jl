module DashSVD

export check_mkl
export dash_svd
include("EigSVD.jl")

using LinearAlgebra, SparseArrays, MKLSparse, MKL, Random

const default_p_max = 1000
const default_tol = 1e-2
Random.seed!(777)

"""
    dash_svd(A, k[, p_max, s, tol]) 

A Julia implementation of the dynamic shifts-based randomized SVD (dashSVD) 
with PVE accuracy control for a matrix `A` of rank `k`.

Parameters:

* `A`:  the input matrix of size (m, n)
* `k`:  the target rank of truncated SVD, k ≤ min(m,n)
* `p_max`: the upper bound of power parameter `p`, default = 1000
* `s`:  the oversampling parameter, **min(m,n) ≥ k + s**, default = `k/2`
* `tol`:the error tolerance for PVE, default = 1e-2

Returns:

* `U`: the matrix of size (m, k) 
       containing the first `k` left singular vectors of `A`
* `S`: the vector of size (k, ) 
       containing the `k` largest singular values of `A` in **ascending** order
* `V`: the matrix of size (n, k) 
       containing the first `k` right singular vectors of `A`



# Examples

```julia
julia> A = randn(10, 6)
julia> U, S, V = dash_svd(A, 2)
```


"""
function dash_svd(A::AbstractMatrix, 
	k::Int, p_max::Int = default_p_max, s::Number = k ÷ 2, 
	tol::Float64 = default_tol)

	if p_max < 0
		@error "Power parameter p must be no less than 0 !"
	end

	if tol < 0
		@error "Error tolerance tol must be no less than 0 !"
	end

	if s + 1 > k || s == 0
		@error "Oversampling parameter s must be a positive integer that satisfies s <= k - 1 !"
	end

	l = k + s
	m, n = size(A)
	if k > min(m, n)
		@error "Rank k must be no more than min(m, n) !"
	end

	if min(m, n) < l
		@error "Upperbound of rank(A) (got: $(min(m,n))) must be no less than k + s (got: $(l)) !"
	end

	if m >= n
		Q = randn(m, l)
		Q = A' * Q
		Q, _, _ = eig_svd(Q)

		alpha = 0.0
		sk = zeros(l)

		for i in 1:p_max
			Q, S, _ = eig_svd(A' * (A*Q) - alpha*Q)
			sk_now = S .+ alpha
			pve_all = abs.(sk_now-sk) ./ sk_now[s]
			ei = maximum(pve_all[s+1:k])

			if ei < tol break end
			if alpha < S[1]
				alpha = (alpha + S[1])/2
			end
			sk = sk_now
		end

		U, S, V = eig_svd(A * Q)

		ind = s+1:k+s
		U = U[1:end, ind]
		V = Q * V[1:end, ind]
		S = S[ind]

	else
		Q = randn(n, l)
		Q = A * Q
		Q, _, _ = eig_svd(Q)

		alpha = 0.0
		sk = zeros(l)

		for i in 1:p_max
			Q, S, _ = eig_svd(A * (A'*Q) - alpha*Q)
			sk_now = S .+ alpha
			pve_all = abs.(sk_now-sk) ./ sk_now[s]
			ei = maximum(pve_all[s+1:k])

			if ei < tol break end
			if alpha < S[1] 
				alpha = (alpha + S[1])/2
			end
			sk = sk_now
		end

		V, S, U = eig_svd(A' * Q)

		ind = s+1:k+s
		V = V[1:end, ind]
		U = Q * U[1:end, ind]
		S = S[ind]

	end

	return U, S, V
end

end
