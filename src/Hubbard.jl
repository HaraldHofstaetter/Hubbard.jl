module Hubbard

export HubbardGlobalData
export init_hubbard, groundstate, double_occupation

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

    fac :: Float64
    shift :: Float64
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

function init_hubbard(N_s::Int, n_up::Int, n_down::Int, v::Array{Float64,2}, U::Float64)
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
    fac = 1.0
    shift = 0.0
    h =  HubbardGlobalData(N_s, n_up, n_down, N_up, N_down, N_psi, N_nz,
                           v, U, Float64[], spzeros(1,1),
                           tab_up, tab_inv_up, tab_down, tab_inv_down,
                           fac, shift)
    gen_H_diag(h)
    gen_H_upper(h)
    h
end

function double_occupation(h::HubbardGlobalData, psi::Union{Array{Complex{Float64},1},Array{Float64,1}})
    r = 0.0
    for i_up = 1:h.N_up
        psi_up = h.tab_inv_up[i_up]
        for i_down = 1:h.N_down
            i = (i_up-1)*h.N_down + i_down
            psi_down = h.tab_inv_down[i_down]
            r += sum(psi_up .& psi_down) * abs(psi[i])^2
        end
    end
    r
end



import Base.LinAlg: A_mul_B!, issymmetric, checksquare
import Base: eltype, size


function A_mul_B!(Y, h::HubbardGlobalData, B)
    Y[:] = h.H_diag.*B + h.H_upper*B + (B'*h.H_upper)'
    if h.fac!=1.0
        Y[:] *= h.fac
    end    
    if h.shift!=0.0
        Y[:] += h.shift*B
    end
end

size(h::HubbardGlobalData) = (h.N_psi, h.N_psi)
eltype(h::HubbardGlobalData) = Float64
issymmetric(h::HubbardGlobalData) = true
checksquare(h::HubbardGlobalData) = h.N_psi 

function groundstate(h::HubbardGlobalData)
    h.fac = 1.0
    h.shift = 0.0 
    lambda,g = eigs(h, nev=1)
    lambda = lambda[1]
    if lambda>=0
        rho = lambda
        h.fac = -1.0
        h.shift = rho
        lambda,g = eigs(h, nev=1)
        lambda = rho-lambda[1]   
    end    
    h.fac = 1.0
    h.shift = 0.0 
    lambda, g[:,1]
end

end
