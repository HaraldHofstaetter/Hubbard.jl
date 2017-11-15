__precompile__()

module Hubbard

export HubbardGlobalData
export hubbard, set_fac_diag, set_fac_offdiag
export groundstate, energy, double_occupation

mutable struct HubbardGlobalData 
    N_s    :: Int
    n_up   :: Int
    n_down :: Int
    N_up   :: Int
    N_down :: Int
    N_psi  :: Int
    N_nz   :: Int

    v :: Array{Float64, 2}
    U :: Float64
    H_diag :: Array{Float64, 1}
    H_upper ::  SparseMatrixCSC{Float64, Int}

    tab_up       :: Dict{BitArray{1},Int}
    tab_inv_up   :: Array{BitArray{1},1} 
    tab_down     :: Dict{BitArray{1},Int}
    tab_inv_down :: Array{BitArray{1},1}

    fac_diag     :: Float64
    fac_offdiag :: Complex{Float64}
    is_complex   :: Bool
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

function gen_H_upper(h::HubbardGlobalData)
    I = zeros(Int, h.N_nz)
    J = zeros(Int, h.N_nz)
    x = zeros(Float64, h.N_nz)
    n = 0
    for i_up = 1:h.N_up
        psi_up = h.tab_inv_up[i_up]
        psi1_hops_up = differ_by_1_entry(psi_up)
        for i_down = 1:h.N_down
            psi_down = h.tab_inv_down[i_down]
            i = (i_up-1)*h.N_down + i_down
            for (psi_new, hop) in psi1_hops_up
                j = (h.tab_up[psi_new]-1)*h.N_down + i_down
                if j>i && h.v[hop[1], hop[2]]!=0.0
                    n += 1
                    I[n] = i
                    J[n] = j
                    x[n] = h.v[hop[1], hop[2]] #* get_sign_up(psi_new, psi_down, hop)
                end
            end
            psi1_hops_down = differ_by_1_entry(psi_down)
            for (psi_new, hop) in psi1_hops_down
                j = (i_up-1)*h.N_down + h.tab_down[psi_new]
                if j>i && h.v[hop[1], hop[2]]!=0.0
                    n += 1
                    I[n] = i
                    J[n] = j
                    x[n] = h.v[hop[1], hop[2]] #* get_sign_down(psi_up, psi_new, hop)
                end
            end            
        end
    end
    h.H_upper = sparse(I[1:n], J[1:n], x[1:n], h.N_psi, h.N_psi) 
end

function gen_H_diag(h::HubbardGlobalData)
    d = zeros(h.N_psi)
    for i_up = 1:h.N_up 
        psi_up = h.tab_inv_up[i_up]
        x_up = sum([ h.v[k,k] for k=1:h.N_s if psi_up[k] ]) 

        for i_down = 1:h.N_down 
            psi_down = h.tab_inv_down[i_down]
            i = (i_up-1)*h.N_down + i_down
            x = x_up + sum([ h.v[k,k] for k=1:h.N_s if psi_down[k] ]) 
            x += sum([h.U for k=1:h.N_s if psi_down[k] && psi_up[k]])
            d[i] = x
        end
    end
    h.H_diag = d
end

dummy_f(t::Float64) = 1.0+0.0im

function hubbard(N_s::Int, n_up::Int, n_down::Int, v::Array{Float64,2}, U::Float64, f::Function=dummy_f)
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
    h =  HubbardGlobalData(N_s, n_up, n_down, N_up, N_down, N_psi, N_nz,
                           v, U, Float64[], spzeros(1,1),
                           tab_up, tab_inv_up, tab_down, tab_inv_down,
                           1.0, 1.0+0.0im, false)
    gen_H_diag(h)
    gen_H_upper(h)
    h
end

function set_fac_diag(h::HubbardGlobalData, f::Float64)
    h.fac_diag = f
end

function set_fac_offdiag(h::HubbardGlobalData, f::Float64)
    h.is_complex = false
    h.fac_offdiag = f
end

function set_fac_offdiag(h::HubbardGlobalData, f::Complex{Float64})
    h.is_complex = true 
    h.fac_offdiag = f
end
    

function double_occupation(h::HubbardGlobalData, psi::Union{Array{Complex{Float64},1},Array{Float64,1}})
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

import Base.LinAlg: A_mul_B!, issymmetric, checksquare
import Base: eltype, size


function A_mul_B!(Y, h::HubbardGlobalData, B)
    if h.is_complex
        Y[:] = h.fac_diag*(h.H_diag.*B) + h.fac_offdiag*(h.H_upper*B) + (h.fac_offdiag*B'*h.H_upper)'
    else
        f = real(h.fac_offdiag)
        Y[:] = h.fac_diag*(h.H_diag.*B) + f*(h.H_upper*B) + (f*B'*h.H_upper)'
    end
end

size(h::HubbardGlobalData) = (h.N_psi, h.N_psi)
eltype(h::HubbardGlobalData) = h.is_complex?Complex{Float64}:Float64
issymmetric(h::HubbardGlobalData) = !h.is_complex || imag(h.fac_offdiag)==0.0
ishermitian(h::HubbardGlobalData) = true
checksquare(h::HubbardGlobalData) = h.N_psi 

function groundstate(h::HubbardGlobalData)
    lambda,g = eigs(h, nev=1, which=:SR)
    lambda = lambda[1]
    real(lambda[1]), g[:,1]
end

function energy(h::HubbardGlobalData, psi::Union{Array{Complex{Float64},1},Array{Float64,1}})
    T = h.is_complex?Complex{Float64}:eltype(psi)
    psi1 = zeros(T, h.N_psi)
    A_mul_B!(psi1, h, psi)
    real(dot(psi,psi1))
end

end
