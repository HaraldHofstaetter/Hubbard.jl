using Hubbard
using FExpokit

struct CommutatorFreeScheme
    A::Array{Float64,2}
    c::Array{Float64,1}
end

CF2 = CommutatorFreeScheme( ones(1,1), [1/2] )

CF4 = CommutatorFreeScheme(
    [1/4+sqrt(3)/6 1/4-sqrt(3)/6
     1/4-sqrt(3)/6 1/4+sqrt(3)/6],
    [1/2-sqrt(3)/6, 1/2+sqrt(3)/6])


function step!(psi::Array{Complex{Float64},1}, h::HubbardHamiltonian, f::Function, 
               t::Real, dt::Real, scheme, b::Array{Float64,1}, norm_inf::Float64)
    fac_diag_save = h.fac_diag
    fac_offdiag_save = h.fac_offdiag
    matrix_times_minus_i_save = h.matrix_times_minus_i

    J,K = size(scheme.A)
    h.matrix_times_minus_i = false # this is done by expv
    for j=1:J
        fac = sum(scheme.A[j,:].*f.(t+dt*scheme.c))
        set_fac_diag(h, b[j]) 
        set_fac_offdiag(h, fac)

        expv!(psi, dt, h, psi, anorm=norm_inf, 
              matrix_times_minus_i=true, hermitian=true,
              wsp=h.wsp, iwsp=h.iwsp) 

    end
    
    h.fac_diag = fac_diag_save
    h.fac_offdiag = fac_offdiag_save
    h.matrix_times_minus_i = matrix_times_minus_i_save
end    

struct EquidistantTimeStepper
    h::HubbardHamiltonian
    f::Function
    psi::Array{Complex{Float64},1}
    t0::Real
    tend::Real
    dt::Real
    scheme
    b::Array{Float64}
    norm_inf::Float64
    function EquidistantTimeStepper(h::HubbardHamiltonian, f::Function, 
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real; scheme=CF4)
        fac_diag_save = h.fac_diag
        fac_offdiag_save = h.fac_offdiag
        matrix_times_minus_i_save =  h.matrix_times_minus_i

        b = [sum(scheme.A[j,:]) for j=1:size(scheme.A,1)]

        h.fac_diag = 1.0
        h.fac_offdiag = 1.0
        h.matrix_times_minus_i = false 
    
        norm_inf = norm(h, Inf)
        
        h.fac_diag = fac_diag_save
        h.fac_offdiag = fac_offdiag_save
        h.matrix_times_minus_i = matrix_times_minus_i_save

        # allocate workspace
        lwsp, liwsp = get_lwsp_liwsp_expv(size(h, 2))  
        h.wsp = zeros(Complex{Float64}, lwsp)
        h.iwsp = zeros(Int32, liwsp)
        
        new(h, f, psi, t0, tend, dt, scheme, b, norm_inf)
    end
end

Base.start(ets::EquidistantTimeStepper) = ets.t0

function Base.done(ets::EquidistantTimeStepper, t) 
    if (t >= ets.tend)
        # deallocate workspace
        ets.h.wsp = Complex{Float64}[]        
        ets.h.iwsp =  Int32[]
        return true
    end
    false
end

function Base.next(ets::EquidistantTimeStepper, t)
    step!(ets.psi, ets.h, ets.f, t, ets.dt, ets.scheme, ets.b, ets.norm_inf)
    t1 = t + ets.dt < ets.tend ? t + ets.dt : ets.tend
    t1, t1
end


function Gamma4!(h::HubbardHamiltonian,
                 r::Vector{Complex{Float64}}, u::Vector{Complex{Float64}}, 
                 t::Float64, b::Float64, f::Complex{Float64}, fd::Complex{Float64}
    set_fac_diag(h, b) 
    A = fd
    B = f
    n = size(h, 2)
    s1 = unsafe_wrap(Array, pointer(h.wsp, 1),   n, own=false)
    s2 = unsafe_wrap(Array, pointer(h.wsp, n+1), n, own=false)

    # s1 = B*u
      set_fac_offdiag(h, B)
      A_mul_B!(s1, h, u)
    # r = c_B*s1, c_B=1 
      r[:] = s1[:] # copy
    # s2 = A*u
      set_fac_offdiag(h, A)
      A_mul_B!(s2, h, u)
    # r += c_A*s2, c_A=t 
      BLAS.axpy!(t, s2, r)
    # s2 = B*s2
      set_fac_offdiag(h, B)
      A_mul_B!(s2, h, s2)
    # r += c_BA*s2, c_BA=-1/2*t^2 
      BLAS.axpy!(-t^2/2, s2, r)
    # s2 = B*s2
      set_fac_offdiag(h, B)
      A_mul_B!(s2, h, s2)
    # r += c_BBA*s2, c_BBA=1/6*t^3
      BLAS.axpy!(t^3/6, s2, r)
    # s2 = B*s2
      set_fac_offdiag(h, B)
      A_mul_B!(s2, h, s2)
    # r += c_BBBA*s2, c_BBBA=1/24*t^4
      BLAS.axpy!(t^4/24, s2, r)

    # s2 = A*s1
      set_fac_offdiag(h, A)
      A_mul_B!(s2, h, s1)
    # r += c_AB*s2, c_AB=1/2*t^2
      BLAS.axpy!(t^2/2, s2, r)
    # s2 = B*s2
      set_fac_offdiag(h, B)
      A_mul_B!(s2, h, s2)
    # r += c_BAB*s2, c_BAB=-1/3*t^3
      BLAS.axpy!(-t^3/3, s2, r)
    # s2 = B*s2
      set_fac_offdiag(h, B)
      A_mul_B!(s2, h, s2)
    # r += c_BBAB*s2, c_BBAB=-1/8*t^4
      BLAS.axpy!(-t^4/8, s2, r)

    # s2 = B*s1
      set_fac_offdiag(h, B)
      A_mul_B!(s2, h, s1)
    # s1 = A*s2
      set_fac_offdiag(h, A)
      A_mul_B!(s1, h, s2)
    # r += c_ABB*s1, c_ABB=1/6t^3 
      BLAS.axpy!(t^3/6, s1, r)
    # s1 = B*s1
      set_fac_offdiag(h, B)
      A_mul_B!(s1, h, s1)
    # r += c_BABB*s1, c_BABB=1/8t^4
      BLAS.axpy!(t^4/8, s1, r)

    # s1 = B*s2
      set_fac_offdiag(h, B)
      A_mul_B!(s1, h, s2)
    # s1 = A*s1
      set_fac_offdiag(h, A)
      A_mul_B!(s1, h, s1)
    # r += c_ABBB*s1, c_ABBB=-1/24*t^4 
      BLAS.axpy!(-t^4/24, s1, r)

end
