

herm(X) = (X+X')/2
is_unitary(X) = (isapprox(X*X', I) && isapprox(X'*X, I))
is_isometry(X) = (isapprox(X*X', I) || isapprox(X'*X, I))

@doc """ random_2qubit_gate()
    
    Produce a random 2-qubit gate (4x4 unitary matrix). May be Haar distributed,
    low entangling, with or without local rotations, depending on the kwargs.
"""
function random_2qubit_gate(; kwargs...)::Matrix
    low_entangling::Bool = get(kwargs, :low_entangling, false)
    if !low_entangling
        return random_nqubit_unitary(2)
    else
        entangling_angle::Float64= get(kwargs, :entangling_angle, .05)
        with_local_rot::Bool = get(kwargs, :with_local_rot, false)
        return low_entangling_matrix(;entangling_angle=entangling_angle, with_local_rot=with_local_rot)
    end
end

@doc """ random_unitary_matrix(n::Integer)
    
    Produce a nxn random unitary matrix. 
    Based on the QR decomposition. Procedure according to 
    https://pennylane.ai/qml/demos/tutorial_haar_measure.html
    WARNING: not 100% confident this is correct, should be checked or replaced 
    at some point
    See also https://math.mit.edu/~edelman/publications/random_matrix_theory.pdf, p20
"""
function random_unitary_matrix(n::Int64)::Matrix{ComplexF64}
    # Generating a complex-valued random matrix of dimensions n x n
    Z = randn(ComplexF64, (n, n))
    # Computing the QR decomposition
    Q, R = qr(Z)
    Lambda = diagm(diag(R)./abs.(diag(R)))
    Haar_random_matrix = Q * Lambda
    return Haar_random_matrix
end

@doc """ Haar_random_vector(n::Integer)
    
    Produce a nxn random unitary matrix. 
    Based on the QR decomposition. Procedure according to 
    https://pennylane.ai/qml/demos/tutorial_haar_measure.html
    Note: reduced computation by only saving the first column of Q and element 
    of R since we are only interested in the first column of the final matrix.
    WARNING: not 100% confident this is correct, should be checked or replaced 
    at some point
    See also https://math.mit.edu/~edelman/publications/random_matrix_theory.pdf, p20
"""
function Haar_random_vector(n::Integer)
    # Generating a complex-valued random matrix of dimensions n x n
    Z = rand(ComplexF64, (n, n))
    # Computing the QR decomposition
    Q, R = qr(Z)
    return (R[1,1] / abs(R[1,1])) .* Q[:, 1]
end
Haar_random_qubit_state(n_qubit::Integer) = Haar_random_vector(2^n_qubit)

random_nqubit_unitary(n_qubits::Int64)::Matrix{ComplexF64} = random_unitary_matrix(2^n_qubits)

function low_entangling_matrix(; entangling_angle::Float64=.05, with_local_rot::Bool=false)::Matrix{ComplexF64}
    # generate random hermitian matrix
    H = random_hermitian_matrix(4)
    # generate the slightly entangling gate
    U_ent = exp(-1im * entangling_angle * H)
    if with_local_rot
        # generate local rotations
        U_rot = kron(random_nqubit_unitary(1), random_nqubit_unitary(1))
        # generate tensor for the unitary gate (exponentiated hermitian)
        return U_rot * U_ent
    else
        # generate tensor for the unitary gate (exponentiated hermitian)
        return U_ent
    end
end

@doc """ random_hermitian_matrix(n::Integer)
    
    Produce a nxn random hermitian matrix. Each entry of the upper triangular 
    part has real and imaginary parts drawn from a normal distribution (0,1), 
    each element of the diagonal is real and normally distributed, while the 
    lower triangular part is the hermitian conjugate of the upper triangular.
"""
function random_hermitian_matrix(n::Int64)::Matrix{ComplexF64}
    H = randn(ComplexF64, (n, n))
    for i in 1:n
        H[i,i] = real(H[i,i])
        for j in 1:i-1
            H[i,j] = conj(H[j,i])
        end
    end
    return H
end

PauliX = [0 1+0im; 1 0]
PauliY = [0 -1im; 1im 0]
PauliZ = [1+0im 0; 0 -1]