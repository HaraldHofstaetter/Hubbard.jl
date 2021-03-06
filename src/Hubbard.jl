__precompile__()

module Hubbard

export HubbardHamiltonian
export hubbard, set_fac!, set_fac_diag!, set_fac_offdiag!
export save_state, restore_state!, set_matrix_times_minus_i! 
export get_norm0, get_wsp, get_iwsp, set_wsp!, set_iwsp!
export groundstate, energy, double_occupation
export get_dims

mutable struct HubbardHamiltonian 
    N_s    :: Int
    n_up   :: Int
    n_down :: Int
    N_up   :: Int
    N_down :: Int
    N_psi  :: Int
    N_nz   :: Int

    v_symm      :: Array{Float64, 2}
    v_anti      :: Array{Float64, 2}
    U      :: Float64
    H_diag :: Array{Float64, 1}
    H_upper_symm ::  SparseMatrixCSC{Float64, Int}
    H_upper_anti ::  SparseMatrixCSC{Float64, Int}

    tab_up       :: Dict{BitArray{1},Int}
    tab_inv_up   :: Array{BitArray{1},1} 
    tab_down     :: Dict{BitArray{1},Int}
    tab_inv_down :: Array{BitArray{1},1}

    fac_diag     :: Float64
    fac_offdiag  :: Complex{Float64}
    matrix_times_minus_i :: Bool
    store_full_matrices  :: Bool 

    wsp  :: Array{Complex{Float64},1}  # workspace for expokit
    iwsp :: Array{Int32,1}    # workspace for expokit
    norm0 :: Float64  # Inf-Norm of H for fac_diag = fac_offdiag = 1, needed by expokit
end


using Combinatorics

function comb_to_bitarray(N::Int, a::Array{Int,1})
    b = falses(N)
    for aa in a
        b[aa] = true
    end
    BitArray(b)   
end

bitarray_to_comb(psi::BitArray{1}) = [k for k=1:length(psi) if psi[k]]

function gen_tabs(N::Integer, n::Integer)
    tab = Dict{BitArray{1},Int}()
    psi0 = falses(N)
    tab_inv = [psi0 for k=1:binomial(N, n)]
    j=0
    for a in Combinatorics.Combinations(1:N,n)
        j+=1
        psi = comb_to_bitarray(N, a)
        tab[psi] = j
        tab_inv[j] = psi
    end
    return tab, tab_inv
end

function differ_by_1_entry(psi::BitArray{1})
    N = length(psi)
    a1 = [k for k=1:N if psi[k]]
    a0 = [k for k=1:N if !psi[k]]
    n1 = length(a1)
    n0 = length(a0)
    psi0 = falses(N) 
    psi1_hops = [(psi0, (0,0)) for k=1:n0*n1]
    j = 0
    for i1=1:n1
        for i0=1:n0
            j+=1
            psi_new = copy(psi)
            psi_new[a0[i0]] = true
            psi_new[a1[i1]] = false            
            psi1_hops[j] = (psi_new, (a1[i1], a0[i0]))
        end
    end
    psi1_hops
end

function get_sign_up(psi_up::BitArray{1}, psi_down::BitArray{1}, hop::Tuple{Int, Int})
    a = minimum(hop)
    b = maximum(hop)
    s = sum(psi_up[a+1:b-1]) + sum(psi_down[a:b-1])
    isodd(s) ? -1 : +1
end

function get_sign_down(psi_up::BitArray{1}, psi_down::BitArray{1}, hop::Tuple{Int, Int})
    a = minimum(hop)
    b = maximum(hop)
    s = sum(psi_up[a+1:b]) + sum(psi_down[a+1:b-1])
    isodd(s) ? -1 : +1
end


function get_dims(H::HubbardHamiltonian)
    i_up = 1
    psi_up = H.tab_inv_up[i_up]
    psi1_hops_up = differ_by_1_entry(psi_up)
    nn_down = zeros(Int, H.N_down)
    for i_down = 1:H.N_down
        psi_down = H.tab_inv_down[i_down]
        i = (i_up-1)*H.N_down + i_down
        for (psi_new, hop) in psi1_hops_up
            j = (H.tab_up[psi_new]-1)*H.N_down + i_down
            if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                nn_down[i_down] += 1
            end
        end
        psi1_hops_down = differ_by_1_entry(psi_down)
        for (psi_new, hop) in psi1_hops_down
            j = (i_up-1)*H.N_down + H.tab_down[psi_new]
            if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                nn_down[i_down] += 1
            end
        end            
    end

    i_down = 1
    psi_down = H.tab_inv_down[i_down]
    psi1_hops_down = differ_by_1_entry(psi_down)
    nn_up = zeros(Int, H.N_up)
    for i_up = 1:H.N_up
        psi_up = H.tab_inv_up[i_up]
        i = (i_down-1)*H.N_up + i_up
        for (psi_new, hop) in psi1_hops_down
            j = (H.tab_down[psi_new]-1)*H.N_up + i_up
            if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                nn_up[i_up] += 1
            end
        end
        psi1_hops_up = differ_by_1_entry(psi_up)
        for (psi_new, hop) in psi1_hops_up
            j = (i_down-1)*H.N_up + H.tab_up[psi_new]
            if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                nn_up[i_up] += 1
            end
        end            
    end
    vcat(0, cumsum([ sum(nn_down-(nn_down[1]-n)) for n in nn_up]))
end

function gen_H_upper_step(i_up::Int, H::HubbardHamiltonian, nn, I, J, x_symm, x_anti)
    n = nn[i_up]
    psi_up = H.tab_inv_up[i_up]
    psi1_hops_up = differ_by_1_entry(psi_up)
    for i_down = 1:H.N_down
        psi_down = H.tab_inv_down[i_down]
        i = (i_up-1)*H.N_down + i_down
        for (psi_new, hop) in psi1_hops_up
            j = (H.tab_up[psi_new]-1)*H.N_down + i_down
            if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                n += 1
                I[n] = i
                J[n] = j
		#s = get_sign_up(psi_new, psi_down, hop)
                x_symm[n] = H.v_symm[hop[1], hop[2]] #*s
                x_anti[n] = H.v_anti[hop[1], hop[2]] #*s
            end
        end
        psi1_hops_down = differ_by_1_entry(psi_down)
        for (psi_new, hop) in psi1_hops_down
            j = (i_up-1)*H.N_down + H.tab_down[psi_new]
            if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                n += 1
                I[n] = i
                J[n] = j
		#s = get_sign_down(psi_up, psi_new, hop)
                x_symm[n] = H.v_symm[hop[1], hop[2]] #*s
                x_anti[n] = H.v_anti[hop[1], hop[2]] #*s
            end
        end            
    end
end

function gen_H_upper_parallel(H::HubbardHamiltonian)
    nn = get_dims(H)
    I = SharedArray{Int,1}(H.N_nz)
    J = SharedArray{Int,1}(H.N_nz)
    x_symm = SharedArray{Float64,1}(H.N_nz)
    x_anti = SharedArray{Float64,1}(H.N_nz)
    pmap(n->gen_H_upper_step(n, H, nn, I, J, x_symm, x_anti), 1:H.N_down) 
    nz = 0
    for n=1:nn[end]
        if x_symm[n]!=0.0
            nz += 1
            I[nz] = I[n]
            J[nz] = J[n]
            x_symm[nz] = x_symm[n]
            x_anti[nz] = x_anti[n]
        end
    end
    H.H_upper_symm = sparse(I[1:nz], J[1:nz], x_symm[1:nz], H.N_psi, H.N_psi) 
    H.H_upper_anti = sparse(I[1:nz], J[1:nz], x_anti[1:nz], H.N_psi, H.N_psi) 
end

function gen_H_diag_parallel(H::HubbardHamiltonian)
    d = SharedArray{Float64,1}(H.N_psi)
    @parallel for i_up = 1:H.N_up 
        psi_up = H.tab_inv_up[i_up]
        x_up = sum([ H.v_symm[k,k] for k=1:H.N_s if psi_up[k] ]) 

        for i_down = 1:H.N_down 
            psi_down = H.tab_inv_down[i_down]
            i = (i_up-1)*H.N_down + i_down
            x = x_up + sum([ H.v_symm[k,k] for k=1:H.N_s if psi_down[k] ]) 
            x += sum([H.U for k=1:H.N_s if psi_down[k] && psi_up[k]])
            d[i] = x
        end
    end
    H.H_diag = sdata(d)
end






function gen_H_upper(H::HubbardHamiltonian)
    I = zeros(Int, H.N_nz)
    J = zeros(Int, H.N_nz)
    x_symm = zeros(Float64, H.N_nz)
    x_anti = zeros(Float64, H.N_nz)
    n = 0
    for i_up = 1:H.N_up
        psi_up = H.tab_inv_up[i_up]
        psi1_hops_up = differ_by_1_entry(psi_up)
        for i_down = 1:H.N_down
            psi_down = H.tab_inv_down[i_down]
            i = (i_up-1)*H.N_down + i_down
            for (psi_new, hop) in psi1_hops_up
                j = (H.tab_up[psi_new]-1)*H.N_down + i_down
                if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                    n += 1
                    I[n] = i
                    J[n] = j
                    #s = get_sign_up(psi_new, psi_down, hop)
                    x_symm[n] = H.v_symm[hop[1], hop[2]] #*s 
                    x_anti[n] = H.v_anti[hop[1], hop[2]] #*s 
                end
            end
            psi1_hops_down = differ_by_1_entry(psi_down)
            for (psi_new, hop) in psi1_hops_down
                j = (i_up-1)*H.N_down + H.tab_down[psi_new]
                if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                    n += 1
                    I[n] = i
                    J[n] = j
                    #s = get_sign_down(psi_up, psi_new, hop)
                    x_symm[n] = H.v_symm[hop[1], hop[2]] #*s
                    x_anti[n] = H.v_anti[hop[1], hop[2]] #*s
                end
            end            
        end
    end
    H.H_upper_symm = sparse(I[1:n], J[1:n], x_symm[1:n], H.N_psi, H.N_psi) 
    H.H_upper_anti = sparse(I[1:n], J[1:n], x_anti[1:n], H.N_psi, H.N_psi) 
end

function gen_H_diag(H::HubbardHamiltonian)
    d = zeros(H.N_psi)
    for i_up = 1:H.N_up 
        psi_up = H.tab_inv_up[i_up]
        x_up = sum([ H.v_symm[k,k] for k=1:H.N_s if psi_up[k] ]) 

        for i_down = 1:H.N_down 
            psi_down = H.tab_inv_down[i_down]
            i = (i_up-1)*H.N_down + i_down
            x = x_up + sum([ H.v_symm[k,k] for k=1:H.N_s if psi_down[k] ]) 
            x += sum([H.U for k=1:H.N_s if psi_down[k] && psi_up[k]])
            d[i] = x
        end
    end
    H.H_diag = d
end


function hubbard(N_s::Int, n_up::Int, n_down::Int, v_symm::Array{Float64,2}, v_anti::Array{Float64,2}, U::Real; 
    store_full_matrices::Bool=false)
    N_up = binomial(N_s, n_up)
    N_down = binomial(N_s, n_down)
    N_psi = N_up*N_down
    N_nz = div(N_psi*(n_up*(N_s-n_up)+n_down*(N_s-n_down)),2)
    tab_up, tab_inv_up = gen_tabs(N_s, n_up)
    if n_up==n_down
        tab_down = tab_up
        tab_inv_down = tab_inv_up
    else
        tab_down, tab_inv_down = gen_tabs(N_s, n_down)
    end
    H =  HubbardHamiltonian(N_s, n_up, n_down, N_up, N_down, N_psi, N_nz,
                           v_symm, v_anti, U, Float64[], spzeros(1,1), spzeros(1,1),
                           tab_up, tab_inv_up, tab_down, tab_inv_down,
                           1.0, 1.0, false, store_full_matrices,
                           Complex{Float64}[], Int32[], 0.0)
    if nprocs()>1
        gen_H_diag_parallel(H)
        gen_H_upper_parallel(H)
    else
        gen_H_diag(H)
        gen_H_upper(H)
    end
    if store_full_matrices
        H.H_upper_symm =  H.H_upper_symm + H.H_upper_symm'
        H.H_upper_anti =  H.H_upper_anti - H.H_upper_anti'
    end
    H.norm0 = norm(H, Inf)
    H
end

get_norm0(H::HubbardHamiltonian) = H.norm0
get_iwsp(H::HubbardHamiltonian) = H.iwsp
get_wsp(H::HubbardHamiltonian) = H.wsp
function set_iwsp!(H::HubbardHamiltonian, n::Integer)  
    H.iwsp = zeros(Int, n) 
end
function set_wsp!(H::HubbardHamiltonian, n::Integer)  
    H.wsp = zeros(Complex{Float64}, n)
end

save_state(H::HubbardHamiltonian) = (H.fac_diag, H.fac_offdiag, H.matrix_times_minus_i) 

function restore_state!(H::HubbardHamiltonian, state)
    (H.fac_diag, H.fac_offdiag, H.matrix_times_minus_i) = state
end

function set_fac!(H::HubbardHamiltonian, fd::Real, fo::Number)
    H.fac_diag = fd
    H.fac_offdiag = fo
end

function set_fac_diag!(H::HubbardHamiltonian, f::Real)
    H.fac_diag = f
end

function set_fac_offdiag!(H::HubbardHamiltonian, f::Number)
    H.fac_offdiag = f
end

function set_matrix_times_minus_i!(H::HubbardHamiltonian, f::Bool)
    H.matrix_times_minus_i = f
end


function double_occupation(H::HubbardHamiltonian, psi::Union{Array{Complex{Float64},1},Array{Float64,1}})
    r = zeros(H.N_s)
    for i_up = 1:H.N_up
        psi_up = H.tab_inv_up[i_up]
        for i_down = 1:H.N_down
            i = (i_up-1)*H.N_down + i_down
            psi_down = H.tab_inv_down[i_down]
            f = abs(psi[i])^2
            for k=1:H.N_s
                if psi_up[k] & psi_down[k]
                    r[k] += f
                end
            end
        end
    end
    r
end

import Base.LinAlg: A_mul_B!, issymmetric, ishermitian, checksquare
import Base: eltype, size, norm


function A_mul_B!(Y, H::HubbardHamiltonian, B)
    fac_symm = real(H.fac_offdiag)
    fac_anti = imag(H.fac_offdiag)

    if H.store_full_matrices
        if fac_anti == 0.0
            Y[:] = H.fac_diag*(H.H_diag.*B) + fac_symm*(H.H_upper_symm*B) 
        else    
            Y[:] = H.fac_diag*(H.H_diag.*B) + fac_symm*(H.H_upper_symm*B) + (1im*fac_anti)*(H.H_upper_anti*B) 
        end
    else
        if fac_anti == 0.0
            Y[:] = H.fac_diag*(H.H_diag.*B) + fac_symm*(H.H_upper_symm*B + At_mul_B(H.H_upper_symm, B)) 
        else    
            Y[:] = H.fac_diag*(H.H_diag.*B) + fac_symm*(H.H_upper_symm*B + At_mul_B(H.H_upper_symm, B)) +  
                                        (1im*fac_anti)*(H.H_upper_anti*B - At_mul_B(H.H_upper_anti, B))
        end
    end
    if H.matrix_times_minus_i
        Y[:] *= -1im
    end

end

size(H::HubbardHamiltonian) = (H.N_psi, H.N_psi)
size(H::HubbardHamiltonian, dim::Int) = dim<1?error("arraysize: dimension out of range"):
                                       (dim<3?H.N_psi:1)
eltype(H::HubbardHamiltonian) = imag(H.fac_offdiag)==0.0?Float64:Complex{Float64}
issymmetric(H::HubbardHamiltonian) = imag(H.fac_offdiag)==0.0 
ishermitian(H::HubbardHamiltonian) = !H.matrix_times_minus_i 
checksquare(H::HubbardHamiltonian) = H.N_psi 

function groundstate(H::HubbardHamiltonian)
    lambda,g = eigs(H, nev=1, which=:SR)
    lambda = lambda[1]
    real(lambda[1]), g[:,1]
end

function energy(H::HubbardHamiltonian, psi::Union{Array{Complex{Float64},1},Array{Float64,1}})
    T = imag(H.fac_offdiag)!=0.0?Complex{Float64}:eltype(psi)
    psi1 = zeros(T, H.N_psi)
    A_mul_B!(psi1, H, psi)
    real(dot(psi,psi1))
end

function norm(H::HubbardHamiltonian, p::Real=2)
    if p==2
        throw(ArgumentError("2-norm not implemented for HubbardHamiltonian. Try norm(H, p) where p=1 or Inf."))
    elseif !(p==1 || p==Inf)
        throw(ArgumentError("invalid p-norm p=$p. Valid: 1, Inf"))
    end
    # TODO consider also H_upper_sym
    s = zeros(H.N_psi)
    for j = 1:H.N_psi
        for i = H.H_upper_symm.colptr[j]:H.H_upper_symm.colptr[j+1]-1
            s[j] += abs(H.H_upper_symm.nzval[i])
        end
    end
    if !H.store_full_matrices
        for i=1:length(H.H_upper_symm.nzval)
            s[H.H_upper_symm.rowval[i]] += abs(H.H_upper_symm.nzval[i])
        end    
    end
    s[:] *= abs(H.fac_offdiag) 
    s[:] += abs(H.fac_diag)*abs.(H.H_diag)
    maximum(s)
end

end
