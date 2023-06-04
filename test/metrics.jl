using LinearAlgebra

function pve_error(A::AbstractMatrix, U::AbstractMatrix, 
					Acc_S::AbstractVector, k::Int, do_reverse::Bool = true)
	if do_reverse == true
		pve = maximum(abs.(reverse(diag(U'*A*(A'*U))) .- Acc_S[1:k].^2))./Acc_S[k+1]^2
	else
		pve = maximum(abs.(diag(U'*A*(A'*U)) .- Acc_S[1:k].^2))./Acc_S[k+1]^2
	end
	return pve
end

function res_error(A::AbstractMatrix, 
					U::AbstractMatrix, S::AbstractVector, V::AbstractMatrix, 
					Acc_S::AbstractVector, k::Int, do_reverse::Bool = true)
	C = A'*U - V*diagm(S)
	if do_reverse
		res = maximum(sqrt.(reverse(diag(C'*C))) ./ Acc_S[1:k])
	else
		res = maximum(sqrt.(diag(C'*C)) ./ Acc_S[1:k])
	end
	return res
end

function sigma_error(S::AbstractVector, Acc_S::AbstractVector, 
					k::Int, do_reverse::Bool = true)
	if do_reverse
		sigma = maximum(abs.(reverse(S) .- Acc_S[1:k]) ./ Acc_S[1:k])
	else
		sigma = maximum(abs.(S .- Acc_S[1:k]) ./ Acc_S[1:k])
	end
	return sigma
end