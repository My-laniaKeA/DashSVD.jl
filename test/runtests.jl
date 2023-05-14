using DashSVD
using Test
using LinearAlgebra, SparseArrays, MKLSparse, MKL

@testset "Check Environment" begin
    @test occursin("mkl_rt", DashSVD.check_mkl())
	@test Threads.nthreads() == 16
end

@testset "Simple Matrix" begin
	A = [	0  0  3;
			1  0  0;
			0  1 -3;
		   -2  0  3;
			0  0  1;	]
	U, S, V = DashSVD.dash_svd(sparse(A), 2)

	S_ = [1.922885871014348; 5.455863287013895]
	@info "S" S
	@test norm(S - S_) < 1e-8
end