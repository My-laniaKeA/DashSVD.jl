module DashSVD

export check_mkl
export dash_svd
include("EigSVD.jl")

using LinearAlgebra, SparseArrays, MKLSparse, MKL, Random

const default_p = 1000
const default_tol = 1e-2
Random.seed!(777)

# [U, S, V] = dash_svd(A, k)
function dash_svd(A::AbstractMatrix, 
	k::Int, p::Int = default_p, s::Number = k ÷ 2, 
	tol::Float64 = default_tol)

	if p < 0
		@warn "Power parameter p must be no less than 0 !"
		return
	end

	if s + 1 > k || s == 0
		@warn "Oversampling parameter s must be a positive integer that satisfies s <= k - 1 !"
		return
	end

	l = k + s
	m, n = size(A)
	if k > min(m, n)
		@warn "Rank k must be no more than min(m, n) !"
		return
	end

	if min(m, n) < l
		@warn "Upperbound of rank(A) (got: $(min(m,n))) must be no less than k + s (got: $(l)) !"
		return
	end

	if m >= n
		Q = randn(m, l)
		Q = A' * Q
		Q, _, _ = eig_svd(Q)

		alpha = 0
		sk_now = 0
		sk = zeros(l)

		for i in 1:p
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

		alpha = 0
		sk_now = 0
		sk = zeros(l)

		for i in 1:p
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
