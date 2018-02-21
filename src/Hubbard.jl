__precompile__()

module Hubbard

export HubbardHamiltonian
export hubbard, set_fac_diag!, set_fac_offdiag!
export save_state, restore_state!
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

    wsp  :: Array{Complex{Float64},1}  # workspace for expokit
    iwsp :: Array{Int32,1}    # workspace for expokit
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


function get_dims(h::HubbardHamiltonian)
    i_up = 1
    psi_up = h.tab_inv_up[i_up]
    psi1_hops_up = differ_by_1_entry(psi_up)
    nn_down = zeros(Int, h.N_down)
    for i_down = 1:h.N_down
        psi_down = h.tab_inv_down[i_down]
        i = (i_up-1)*h.N_down + i_down
        for (psi_new, hop) in psi1_hops_up
            j = (h.tab_up[psi_new]-1)*h.N_down + i_down
            if j>i && h.v_symm[hop[1], hop[2]]!=0.0
                nn_down[i_down] += 1
            end
        end
        psi1_hops_down = differ_by_1_entry(psi_down)
        for (psi_new, hop) in psi1_hops_down
            j = (i_up-1)*h.N_down + h.tab_down[psi_new]
            if j>i && h.v_symm[hop[1], hop[2]]!=0.0
                nn_down[i_down] += 1
            end
        end            
    end

    i_down = 1
    psi_down = h.tab_inv_down[i_down]
    psi1_hops_down = differ_by_1_entry(psi_down)
    nn_up = zeros(Int, h.N_up)
    for i_up = 1:h.N_up
        psi_up = h.tab_inv_up[i_up]
        i = (i_down-1)*h.N_up + i_up
        for (psi_new, hop) in psi1_hops_down
            j = (h.tab_down[psi_new]-1)*h.N_up + i_up
            if j>i && h.v_symm[hop[1], hop[2]]!=0.0
                nn_up[i_up] += 1
            end
        end
        psi1_hops_up = differ_by_1_entry(psi_up)
        for (psi_new, hop) in psi1_hops_up
            j = (i_down-1)*h.N_up + h.tab_up[psi_new]
            if j>i && h.v_symm[hop[1], hop[2]]!=0.0
                nn_up[i_up] += 1
            end
        end            
    end
    vcat(0, cumsum([ sum(nn_down-(nn_down[1]-n)) for n in nn_up]))
end

function gen_H_upper_step(i_up::Int, h::HubbardHamiltonian, nn, I, J, x_symm, x_anti)
    n = nn[i_up]
    psi_up = h.tab_inv_up[i_up]
    psi1_hops_up = differ_by_1_entry(psi_up)
    for i_down = 1:h.N_down
        psi_down = h.tab_inv_down[i_down]
        i = (i_up-1)*h.N_down + i_down
        for (psi_new, hop) in psi1_hops_up
            j = (h.tab_up[psi_new]-1)*h.N_down + i_down
            if j>i && h.v_symm[hop[1], hop[2]]!=0.0
                n += 1
                I[n] = i
                J[n] = j
		#s = get_sign_up(psi_new, psi_down, hop)
                x_symm[n] = h.v_symm[hop[1], hop[2]] #*s
                x_anti[n] = h.v_anti[hop[1], hop[2]] #*s
            end
        end
        psi1_hops_down = differ_by_1_entry(psi_down)
        for (psi_new, hop) in psi1_hops_down
            j = (i_up-1)*h.N_down + h.tab_down[psi_new]
            if j>i && h.v_symm[hop[1], hop[2]]!=0.0
                n += 1
                I[n] = i
                J[n] = j
		#s = get_sign_down(psi_up, psi_new, hop)
                x_symm[n] = h.v_symm[hop[1], hop[2]] #*s
                x_anti[n] = h.v_anti[hop[1], hop[2]] #*s
            end
        end            
    end
end

function gen_H_upper_parallel(h::HubbardHamiltonian)
    nn = get_dims(h)
    I = SharedArray{Int,1}(h.N_nz)
    J = SharedArray{Int,1}(h.N_nz)
    x_symm = SharedArray{Float64,1}(h.N_nz)
    x_anti = SharedArray{Float64,1}(h.N_nz)
    pmap(n->gen_H_upper_step(n, h, nn, I, J, x_symm, x_anti), 1:h.N_down) 
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
    h.H_upper_symm = sparse(I[1:nz], J[1:nz], x_symm[1:nz], h.N_psi, h.N_psi) 
    h.H_upper_anti = sparse(I[1:nz], J[1:nz], x_anti[1:nz], h.N_psi, h.N_psi) 
end

function gen_H_diag_parallel(h::HubbardHamiltonian)
    d = SharedArray{Float64,1}(h.N_psi)
    @parallel for i_up = 1:h.N_up 
        psi_up = h.tab_inv_up[i_up]
        x_up = sum([ h.v_symm[k,k] for k=1:h.N_s if psi_up[k] ]) 

        for i_down = 1:h.N_down 
            psi_down = h.tab_inv_down[i_down]
            i = (i_up-1)*h.N_down + i_down
            x = x_up + sum([ h.v_symm[k,k] for k=1:h.N_s if psi_down[k] ]) 
            x += sum([h.U for k=1:h.N_s if psi_down[k] && psi_up[k]])
            d[i] = x
        end
    end
    h.H_diag = sdata(d)
end






function gen_H_upper(h::HubbardHamiltonian)
    I = zeros(Int, h.N_nz)
    J = zeros(Int, h.N_nz)
    x_symm = zeros(Float64, h.N_nz)
    x_anti = zeros(Float64, h.N_nz)
    n = 0
    for i_up = 1:h.N_up
        psi_up = h.tab_inv_up[i_up]
        psi1_hops_up = differ_by_1_entry(psi_up)
        for i_down = 1:h.N_down
            psi_down = h.tab_inv_down[i_down]
            i = (i_up-1)*h.N_down + i_down
            for (psi_new, hop) in psi1_hops_up
                j = (h.tab_up[psi_new]-1)*h.N_down + i_down
                if j>i && h.v_symm[hop[1], hop[2]]!=0.0
                    n += 1
                    I[n] = i
                    J[n] = j
                    #s = get_sign_up(psi_new, psi_down, hop)
                    x_symm[n] = h.v_symm[hop[1], hop[2]] #*s 
                    x_anti[n] = h.v_anti[hop[1], hop[2]] #*s 
                end
            end
            psi1_hops_down = differ_by_1_entry(psi_down)
            for (psi_new, hop) in psi1_hops_down
                j = (i_up-1)*h.N_down + h.tab_down[psi_new]
                if j>i && h.v_symm[hop[1], hop[2]]!=0.0
                    n += 1
                    I[n] = i
                    J[n] = j
                    #s = get_sign_down(psi_up, psi_new, hop)
                    x_symm[n] = h.v_symm[hop[1], hop[2]] #*s
                    x_anti[n] = h.v_anti[hop[1], hop[2]] #*s
                end
            end            
        end
    end
    h.H_upper_symm = sparse(I[1:n], J[1:n], x_symm[1:n], h.N_psi, h.N_psi) 
    h.H_upper_anti = sparse(I[1:n], J[1:n], x_anti[1:n], h.N_psi, h.N_psi) 
end

function gen_H_diag(h::HubbardHamiltonian)
    d = zeros(h.N_psi)
    for i_up = 1:h.N_up 
        psi_up = h.tab_inv_up[i_up]
        x_up = sum([ h.v_symm[k,k] for k=1:h.N_s if psi_up[k] ]) 

        for i_down = 1:h.N_down 
            psi_down = h.tab_inv_down[i_down]
            i = (i_up-1)*h.N_down + i_down
            x = x_up + sum([ h.v_symm[k,k] for k=1:h.N_s if psi_down[k] ]) 
            x += sum([h.U for k=1:h.N_s if psi_down[k] && psi_up[k]])
            d[i] = x
        end
    end
    h.H_diag = d
end


function hubbard(N_s::Int, n_up::Int, n_down::Int, v_symm::Array{Float64,2}, v_anti::Array{Float64,2}, U::Real)
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
    h =  HubbardHamiltonian(N_s, n_up, n_down, N_up, N_down, N_psi, N_nz,
                           v_symm, v_anti, U, Float64[], spzeros(1,1), spzeros(1,1),
                           tab_up, tab_inv_up, tab_down, tab_inv_down,
                           1.0, 1.0, false, Complex{Float64}[], Int32[])
    if nprocs()>1
        gen_H_diag_parallel(h)
        gen_H_upper_parallel(h)
    else
        gen_H_diag(h)
        gen_H_upper(h)
    end
    h
end

save_state(h::HubbardHamiltonian) = (h.fac_diag, h.fac_offdiag, h.matrix_times_minus_i) 

function restore_state!(h::HubbardHamiltonian, state)
    (h.fac_diag, h.fac_offdiag, h.matrix_times_minus_i) = state
end

function set_fac_diag!(h::HubbardHamiltonian, f::Real)
    h.fac_diag = f
end

function set_fac_offdiag!(h::HubbardHamiltonian, f::Number)
    h.fac_offdiag = f
end


function double_occupation(h::HubbardHamiltonian, psi::Union{Array{Complex{Float64},1},Array{Float64,1}})
    r = zeros(h.N_s)
    for i_up = 1:h.N_up
        psi_up = h.tab_inv_up[i_up]
        for i_down = 1:h.N_down
            i = (i_up-1)*h.N_down + i_down
            psi_down = h.tab_inv_down[i_down]
            f = abs(psi[i])^2
            for k=1:h.N_s
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


function A_mul_B!(Y, h::HubbardHamiltonian, B)
    fac_symm = real(h.fac_offdiag)
    fac_anti = imag(h.fac_offdiag)
    if fac_anti == 0.0
        Y[:] = h.fac_diag*(h.H_diag.*B) + fac_symm*(h.H_upper_symm*B) + (fac_symm*B'*h.H_upper_symm)'
    else    
        Y[:] = h.fac_diag*(h.H_diag.*B) + fac_symm*(h.H_upper_symm*B) + (fac_symm*B'*h.H_upper_symm)' + 
                                     1im*(fac_anti*(h.H_upper_anti*B) - (fac_anti*B'*h.H_upper_anti)')  
    end
    if h.matrix_times_minus_i
        Y[:] *= -1im
    end
end

size(h::HubbardHamiltonian) = (h.N_psi, h.N_psi)
size(h::HubbardHamiltonian, dim::Int) = dim<1?error("arraysize: dimension out of range"):
                                       (dim<3?h.N_psi:1)
eltype(h::HubbardHamiltonian) = imag(h.fac_offdiag)==0.0?Float64:Complex{Float64}
issymmetric(h::HubbardHamiltonian) = imag(h.fac_offdiag)==0.0 
ishermitian(h::HubbardHamiltonian) = !h.matrix_times_minus_i 
checksquare(h::HubbardHamiltonian) = h.N_psi 

function groundstate(h::HubbardHamiltonian)
    lambda,g = eigs(h, nev=1, which=:SR)
    lambda = lambda[1]
    real(lambda[1]), g[:,1]
end

function energy(h::HubbardHamiltonian, psi::Union{Array{Complex{Float64},1},Array{Float64,1}})
    T = imag(h.fac_offdiag)!=0.0?Complex{Float64}:eltype(psi)
    psi1 = zeros(T, h.N_psi)
    A_mul_B!(psi1, h, psi)
    real(dot(psi,psi1))
end

function norm(h::HubbardHamiltonian, p::Real=2)
    if p==2
        throw(ArgumentError("2-norm not implemented for HubbardHamiltonian. Try norm(h, p) where p=1 or Inf."))
    elseif !(p==1 || p==Inf)
        throw(ArgumentError("invalid p-norm p=$p. Valid: 1, Inf"))
    end
    s = zeros(h.N_psi)
    for j = 1:h.N_psi
        for i = h.H_upper_symm.colptr[j]:h.H_upper_symm.colptr[j+1]-1
            s[j] += abs(h.H_upper_symm.nzval[i])
        end
    end
    for i=1:length(h.H_upper_symm.nzval)
        s[h.H_upper_symm.rowval[i]] += abs(h.H_upper_symm.nzval[i])
    end    
    s[:] *= abs(h.fac_offdiag) 
    s[:] += abs(h.fac_diag)*abs.(h.H_diag)
    maximum(s)
end

end
