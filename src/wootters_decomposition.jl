using LinearAlgebra


"""
    Wootters building blocks
    - eigendecomposition -> U1, U2 & U3 (from the paper)
    - eigendecomp -> optimal decomp z = v U3'U2'U1'
    - arbitrary decomp -> Uopt (additional SVD) Uopt = V_phi U3'U2'U1'
"""
function wootters(decomposition_tensor::Array{ComplexF64, 3}; do_checks::Bool=false)
    dout, din, dk = size(decomposition_tensor)
    @assert dout==din
    @assert din==2
    U_opt, final_decomp = wootters(reshape(decomposition_tensor, 4,dk); do_checks=do_checks)
    return reshape(final_decomp, 2,2,size(final_decomp)[2])
end
function wootters(decomposition_matrix::Matrix{ComplexF64}; do_checks::Bool=false)
    # Initial eigendecomposition through SVD
    F = svd(decomposition_matrix)
    v = F.U * Diagonal(F.S)             # sub-normalized the eigen-decomposition matrix
    U_0 = F.Vt                          # unitary mapping the input decomposition to the eigendecomposition
    if do_checks
        @assert all(>(0), F.S)
        density_matrix = decomposition_matrix * decomposition_matrix'
        @assert norm(F.U * Diagonal(F.S.^2) * F.U' - density_matrix) < sqrt(eps())           # checking eigendecomposition
        @assert norm(v * v' - density_matrix) < sqrt(eps()) 
    end
    Ux, x, lambda_R = wootters_first_unitary(v; do_checks=do_checks)
    if is_separable(lambda_R; tol=eps())
        Uz, z = wootters_separable_unitary(x, lambda_R; do_checks=do_checks)
        U = Uz * Ux
    else
        Uy, y = wootters_second_unitary(x; do_checks=do_checks, R_eigenvalues=lambda_R)
        Uz, z = wootters_third_unitary(y; do_checks=do_checks)
        if do_checks
            Z = z' * spin_flipped_state_vector(z)
            Z = round.(Z, digits=14)
            p_Z = z' * z
            C_rho = concurrence(lambda_R)
            for i in 1:4
                # @show Z[i,i]/p_Z[i,i], C_rho
                # @assert isapprox(Z[i,i]/p_Z[i,i], C_rho)
                @assert abs(Z[i,i]/p_Z[i,i] - C_rho) < 1e-5 "Z[i,i]/p_Z[i,i] = $(Z[i,i]/p_Z[i,i]), C_rho = $(C_rho), abs(Z[i,i]/p_Z[i,i] - C_rho) = $(abs(Z[i,i]/p_Z[i,i] - C_rho)) > 0"
            end
        end
        U = Uz * Uy * Ux * U_0
    end
    return U, z
end


function wootters_first_unitary(eigen_decomp::Matrix{ComplexF64}; do_checks::Bool=false)
    v_subnorm = eigen_decomp
    tau = v_subnorm' * spin_flipped_state_vector(v_subnorm)
    tau = isapprox(tau, real(tau)) ? real(tau) : tau
    tau = round.((tau+transpose(tau))/2, digits=14)
    S, W = takagi(tau)
    Ux = W'
    x = v_subnorm * Ux'
    if do_checks
        density_matrix = v_subnorm * v_subnorm'
        @assert isapprox(x * x', density_matrix)
        #TODO: check why the difference between S and the eigvals of R can be "so large"
        @assert norm(S - R_eigenvalues(density_matrix)[1:length(S)]) < 1e-6 "AssertionError: norm(S - R_eigenvalues(density_matrix)[1:length(S)]) < sqrt(eps()) : S = $(S) but R_eigenvalues(density_matrix) = $(R_eigenvalues(density_matrix))"
        @assert norm(R_eigenvalues(density_matrix)[length(S)+1:end]) < sqrt(eps())
        @assert isapprox(tau, transpose(tau), rtol=eps())                               # checking symmetric
        @assert norm(tau - transpose(tau)) < eps()                                      # checking symmetric
        D = Diagonal(S)
        @assert norm(Ux * tau * transpose(Ux) - D) < sqrt(eps())            # checking Autonne-Takagi decomposition
        @assert norm(tau - Ux' * D * conj(Ux)) < sqrt(eps())                # checking Autonne-Takagi decomposition
        X = x' * spin_flipped_state_vector(x)
        @assert norm(X - Diagonal(X)) < sqrt(eps())                                 # checking diagonal
    end
    return Ux, x, S
end

function wootters_second_unitary(first_decomp::Matrix{ComplexF64}; do_checks::Bool=false, R_eigenvalues::Array{<:Real})
    n_elements = size(first_decomp)[2]
    Uy = Diagonal(vcat([1], 1im*ones(n_elements-1)))
    y = first_decomp * Uy'
    if do_checks
        # TODO assert that it is an entangled case
        # TODO assert that the diagonal of first_decomp' * spin_flipped_state_vector(first_decomp) is the same as R_eigenvalues
        Y = y' * spin_flipped_state_vector(y)
        if (C_rho = concurrence(R_eigenvalues)) > 0           # should always be the case!
            @assert isapprox(y * y', first_decomp * first_decomp')
            # @show Y
            # @show C_rho, tr(Y)
            @assert norm(Y - Diagonal(Y)) < sqrt(eps())
            # @assert isapprox(tr(Y), C_rho)
            @assert abs(tr(Y) - C_rho) < 1e-7 "tr(Y) = $(tr(Y)), C_rho = $C_rho, abs(tr(Y) - C_rho) = $(abs(tr(Y) - C_rho)) > 0"
        else
            @assert abs(tr(Y)) < sqrt(eps()) "abs(tr(Y)) = $(abs(tr(Y))) > sqrt(eps()) > 0"
        end
    end
    return Uy, y
end

function wootters_third_unitary(second_decomp::Matrix{ComplexF64}; do_checks::Bool=false)
    n_elements = size(second_decomp)[2]
    Y = second_decomp' * spin_flipped_state_vector(second_decomp)
    C_rho = tr(Y)
    P_Y = second_decomp' * second_decomp
    M = Y - C_rho * P_Y
    _, Q_M = eigen(Hermitian(real(M)))
    H4 = (1/2) * [1 1 1 1; 1 1 -1 -1; 1 -1 1 -1; 1 -1 -1 1]
    Uz = H4[:,1:n_elements] * Q_M'
    z = second_decomp * Uz'
    if do_checks
        @assert isapprox(z * z', second_decomp * second_decomp')
        @assert isapprox(tr(P_Y), 1)
        # @show tr(M)
        # @show lambda_M
        @assert abs(tr(M)) < sqrt(eps())
        mixed_M = Uz * M * Uz'
        # @show diag(H4 * Diagonal(lambda_M) * H4')
        # @show diag(Q_M' * M * Q_M)
        # @show diag(H4 * Q_M' * M * Q_M * H4')
        # @show diag(mixed_M)
        for i in 1:4
            # @show i, mixed_M[i,i], abs(mixed_M[i,i]), eps()
            @assert abs(mixed_M[i,i]) < sqrt(eps()) "abs(mixed_M[$i,$i]) = $(abs(mixed_M[i,i])) > sqrt(eps()) > 0"
        end
        Z = z' * spin_flipped_state_vector(z)
        Z = round.(Z, digits=14)
        P_Z = z' * z
        # @show Z[1,1] , Z[2,2] , Z[3,3] , Z[4,4] , tr(Y)/4
        # @assert isapprox(Z[1,1], tr(Y)/4)
        # @assert isapprox(Z[3,3], tr(Y)/4)
        # @assert isapprox(Z[4,4], tr(Y)/4)
        # @assert isapprox(Z[4,4], tr(Y)/4)
        @assert isapprox(tr(P_Z), 1)
        # @show diag(Z) ./ diag(P_Z)
        for i in 1:4
            # @assert (isapprox(Z[i,i]/P_Z[i,i] , tr(Y)) || abs(Z[i,i]/P_Z[i,i]) < eps())
            # @show Z[i,i]/P_Z[i,i], tr(Y), abs(Z[i,i]/P_Z[i,i] - tr(Y))
            @assert (abs(Z[i,i]/P_Z[i,i] - tr(Y)) < 1e-5 || abs(Z[i,i]/P_Z[i,i]) < 1e-5) "Z[i,i]/P_Z[i,i] = $(Z[i,i]/P_Z[i,i]), tr(Y) = $(tr(Y)), abs(Z[i,i]/P_Z[i,i] - tr(Y)) = $(abs(Z[i,i]/P_Z[i,i] - tr(Y))) > 0 and Z[i,i]/P_Z[i,i] = $(Z[i,i]/P_Z[i,i]) > 0"
        end
    end
    return Uz, z
end

function wootters_separable_unitary(first_decomp::Matrix{ComplexF64}, lambda_R::Array{Float64}; do_checks::Bool=false)
    if length(lambda_R) < 4
        lambda_R = vcat(lambda_R, zeros(typeof(lambda_R[1]), 4-length(lambda_R)))
        first_decomp = cat(first_decomp, zeros(typeof(first_decomp[1]), 4,4-size(first_decomp)[2]); dims=2)
    end
    thetas = trapezoid_angles(lambda_R)
    Uz = (1 / 2) * [
        exp(im*thetas[1]) exp(im*thetas[2]) exp(im*thetas[3]) exp(im*thetas[4]); 
        exp(im*thetas[1]) exp(im*thetas[2]) -exp(im*thetas[3]) -exp(im*thetas[4]); 
        exp(im*thetas[1]) -exp(im*thetas[2]) exp(im*thetas[3]) -exp(im*thetas[4]); 
        exp(im*thetas[1]) -exp(im*thetas[2]) -exp(im*thetas[3]) exp(im*thetas[4])] 
    z = first_decomp * Uz'
    if do_checks
        @assert abs(lambda_R' * exp.(2im*thetas)) < 1e-7
        Z = z' * spin_flipped_state_vector(z)
        @assert abs(tr(Z)) < 1e-7 "abs(tr(Z)) = $(abs(tr(Z))) > 0" 
    end
    return Uz, z
end

YY = reverse(diagm([1, -1, -1, 1]), dims=1)
spin_flipped_density(state::Matrix{ComplexF64}) = YY * conj(state) * YY
spin_flipped_state_vector(state::Union{Vector{ComplexF64}, Matrix{ComplexF64}}) = YY * conj(state)


function R_eigenvalues(state::Matrix{ComplexF64})
    @assert size(state) == (4, 4)
    lambda = eigvals(state * spin_flipped_density(state))
    lambda = real.(lambda)
    @assert isnothing(findfirst(<(-sqrt(eps())), lambda))
    lambda = round.(max.(0, lambda), digits=14)
    lambda = sort(sqrt.(lambda); rev=true)
    return lambda
end

concurrence(R_eigenvalues::Array{<:Real}) = max(0, R_eigenvalues[1] - sum(R_eigenvalues[2:end]))

is_separable(state::Matrix{ComplexF64}; tol::Float64=0.) = is_separable(R_eigenvalues(state); tol=tol)
is_separable(R_eigenvalues::Array{Float64}; tol::Float64=0.) = (R_eigenvalues[1] - sum(R_eigenvalues[2:end]) < tol)

@doc """
    takagi(M::Matrix)

    computing the Autonne-Takagi decomposition of a symmetric matrix M.
    Based on codes from SymplecticDecompositions.jl from Nicolas Quesada
    (https://github.com/polyquantique/SymplecticDecompositions.jl/tree/main)
    and Strawbery Fields (https://strawberryfields.readthedocs.io/en/stable/_modules/strawberryfields/decompositions.html#takagi)
    TODO: finish transcribing the SF code to deal with degeneracies
"""
function takagi(M::Matrix{ComplexF64}; tol::Float64=eps())
    n, m = size(M)
    @assert n == m                                                  # is square
    @assert isapprox(M, transpose(M), rtol=tol)                     # is approximately symmetric

    if all(entry->abs(entry) < tol, M)                              # approximately the zero matrix
        return zeros(n), I
    end
    
    if isapprox(M, real(M))                                                  # approximately a real matrix
        return takagi(real(M); tol=tol)
    else
        u, s, v = svd(M)
        pref = u' * conj(v)
        pref12 = normal_sqrtm(pref)
        return s, u * pref12
    end
end
function takagi(M::Matrix{Float64}; tol::Float64=eps())
    n, m = size(M)
    @assert n == m                                                  # is square
    @assert isapprox(M, transpose(M), rtol=tol)                     # is approximately symmetric

    if all(entry->abs(entry) < tol, M)                              # approximately the zero matrix
        return zeros(n), I
    end

    # If the matrix N is real one can be more clever and use its eigendecomposition
    M = real(M)
    l, U = eigen(Hermitian(M))
    # @show l, U
    vals = abs.(l)          # These are the Takagi eigenvalues
    phases = [(i > 0) ? 1 : 1im for i in l]
    Uc = Matrix{ComplexF64}(U * Diagonal(phases))
    ordering = sortperm(vals; rev=true)
    vals = vals[ordering]
    Uc = Uc[:, ordering]
    # @show vals, Uc
    return vals, Uc
end

function normal_sqrtm(A)
    T, Z = schur(A)
    return Z * sqrt(T) * transpose(conj(Z))
end

function trapezoid_angles(lambdas::Vector{Float64})
    @assert length(lambdas) == 4
    if lambdas[1] == lambdas[2] == lambdas[3] == lambdas[4]
        return [0, pi/4, pi/2, 3*pi/4]
    # elseif lambdas[2] == lambdas[3]
    #     l = (lambdas[1] - lambdas[4])/2
    #     theta = acos(l/lambdas[2])/2
    #     return [0, (pi-theta)/2, (pi+theta)/2, pi/2]
    else
        s = (lambdas[1] - lambdas[4] + lambdas[2] + lambdas[3]) / 2
        h = (2/(lambdas[1] - lambdas[4])) * sqrt(s * (s - lambdas[1] + lambdas[4]) * (s - lambdas[2]) * (s - lambdas[3]))
        theta2 = (pi-asin(h/lambdas[2]))/2
        theta3 = (pi+asin(h/lambdas[3]))/2
        # theta3 = (cos(2*theta2)*lambdas[2] + lambdas[3] > lambdas[1]) ? -asin(h/lambdas[3])/2 : (pi+asin(h/lambdas[3]))/2
        thetas = [0, theta2, theta3, pi/2]
        if abs(lambdas' * exp.(2im*thetas)) > 1e-7
            thetas[3] = -asin(h/lambdas[3])/2
        end
        # @show lambdas
        # @show s, h
        # @show h/lambdas[2], h/lambdas[3]
        # @show asin(h/lambdas[2]), asin(h/lambdas[3])
        # @show sin(asin(h/lambdas[2]))-h/lambdas[2], sin(asin(h/lambdas[3]))-h/lambdas[3]
        # @show theta2, theta3
        # @show lambdas[1] - lambdas[4] - lambdas[2] - lambdas[3]
        # @show lambdas' * exp.(2im*thetas)
        # @show lambdas[1] - lambdas[4] + lambdas[2]*cos(2*theta2) + lambdas[3]*cos(2*theta3)
        # @show lambdas[1] - lambdas[4] - lambdas[2]*sqrt(1-(h/lambdas[2])^2) - lambdas[3]*sqrt(1-(h/lambdas[3])^2)
        # @show lambdas[3]*cos(asin(h/lambdas[3])) + lambdas[4]-lambdas[1]
        return thetas
    end
end

nothing
