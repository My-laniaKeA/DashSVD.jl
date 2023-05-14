using DashSVD
using Test
using LinearAlgebra, SparseArrays, MKLSparse, MKL

@testset "Environment" begin
    @test occursin("mkl_rt", DashSVD.check_mkl())
	@test Threads.nthreads() == 16
end


@testset "Simple Matrix: test dashSVD with m = $m, n = $n, and p = $p" for
    (m, n, p) in ((10, 6, 0.8),
                  (6, 10, 0.8),
                  (10, 10, 0.8),
                  (100, 60, 0.1),
                  (60, 100, 0.1),
                  (100, 100, 0.1))

	mnp = round(Integer, m*n*p)
	tol = 1e-2

	for A in (randn(m, n),
		complex.(randn(m, n), randn(m, n)),
		sprandn(m, n, p),
		sparse(rand(1:m, mnp), rand(1:n, mnp), complex.(randn(mnp), randn(mnp)), m, n))

		Uf, Sf, Vf = svd(Array(A))

		for k = 2:5
			s = k ÷ 2 
			if min(m, n) < k + s	# Violating the requirement of eig_svd
				continue 
			end
			U, S, V = DashSVD.dash_svd(A, k)
			@test norm(S - Sf[k:-1:1], Inf) < tol * mnp
			@test norm(abs.(U'Uf[:,k:-1:1]) - I, Inf) < tol * mnp
			@test norm(abs.(V'Vf[:,k:-1:1]) - I, Inf) < tol * mnp
		end
	end
end