using DashSVD
using Test
using LinearAlgebra, SparseArrays, MKLSparse, MKL
using CSV, DataFrames
using BenchmarkTools

include("metrics.jl")

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
			@test pve_error(A, U, Sf, k) < tol * 10 # for small matrix, it is possible that pve > tol
			@test norm(S - Sf[k:-1:1], Inf) < tol * mnp
			@test norm(abs.(U'Uf[:,k:-1:1]) - I, Inf) < tol * mnp
			@test norm(abs.(V'Vf[:,k:-1:1]) - I, Inf) < tol * mnp
		end
	end
end


@testset "SNAP" begin
	csc_matrix = CSV.read("SNAP.csv", DataFrame)
	colptr = csc_matrix[:,1]
	rowval = csc_matrix[:,2]
	nzval = csc_matrix[:,3]
	A = sparse(colptr, rowval, nzval)
	m, n = size(A)
	new_row = sparse(zeros(1, n))
	A = vcat(A, new_row)
	k = 100
	tol = 1e-2
	U, S, V = @btime DashSVD.dash_svd($A, $k)
	sv = CSV.read("SNAP_sv.csv", DataFrame)
	Acc_S = sv[:,1]	# descending order
	pve = pve_error(A, U, Acc_S, k)
	@info "pve error" pve
	@test pve < tol * 3
	res = res_error(A, U, S, V, Acc_S, k)
	@info "res error" res
	@test res < tol * 3
	sigma = sigma_error(S, Acc_S, k)
	@info "sigma error" sigma
	@test sigma < tol * 3
end