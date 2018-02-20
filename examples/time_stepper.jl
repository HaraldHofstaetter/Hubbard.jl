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

