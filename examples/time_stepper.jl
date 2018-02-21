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
               t::Real, dt::Real, scheme::CommutatorFreeScheme, norm_inf::Float64)
    state = save_state(h)

    J,K = size(scheme.A)
    h.matrix_times_minus_i = false # this is done by expv
    for j=1:J
        set_fac_diag!(h,
            sum(scheme.A[j,:])) 
        set_fac_offdiag!(h, 
            sum(scheme.A[j,:].*f.(t+dt*scheme.c)))

        expv!(psi, dt, h, psi, anorm=norm_inf, 
              matrix_times_minus_i=true, hermitian=true,
              wsp=h.wsp, iwsp=h.iwsp) 

    end

    restore_state!(h, state)
end    

function norm_fac1(h::HubbardHamiltonian)
    state = save_state(h)
    set_fac_diag!(h, 1.0) 
    set_fac_offdiag!(h, 1.0)
    h.matrix_times_minus_i = false 
    r = norm(h, Inf)
    restore_state!(h, state)
    r
end


struct EquidistantTimeStepper
    h::HubbardHamiltonian
    f::Function
    psi::Array{Complex{Float64},1}
    t0::Real
    tend::Real
    dt::Real
    scheme
    norm_inf::Float64
    function EquidistantTimeStepper(h::HubbardHamiltonian, f::Function, 
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real; scheme=CF4)

        norm_inf = norm_fac1(h)
        # allocate workspace
        lwsp, liwsp = get_lwsp_liwsp_expv(size(h, 2))  
        h.wsp = zeros(Complex{Float64}, lwsp)
        h.iwsp = zeros(Int32, liwsp)
        
        new(h, f, psi, t0, tend, dt, scheme, norm_inf)
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
    step!(ets.psi, ets.h, ets.f, t, ets.dt, ets.scheme, ets.norm_inf)
    t1 = t + ets.dt < ets.tend ? t + ets.dt : ets.tend
    t1, t1
end

function local_orders(h::HubbardHamiltonian, f::Function,
                      psi::Array{Complex{Float64},1}, t0::Real, dt::Real; 
                      scheme=CF2, reference_scheme=scheme, 
                      reference_steps=10,
                      rows=8)
    tab = zeros(Float64, rows, 3)

    wf_save_initial_value = copy(psi)
    psi_ref = copy(psi)

    norm_inf = norm_fac1(h)

    dt1 = dt
    err_old = 0.0
    println("             dt         err      p")
    println("-----------------------------------")
    for row=1:rows
        step!(psi, h, f, t0, dt1, scheme, norm_inf)
        psi_ref = copy(wf_save_initial_value)
        dt1_ref = dt1/reference_steps
        for k=1:reference_steps
            step!(psi_ref, h, f, t0+(k-1)*dt1_ref, dt1_ref, reference_scheme, norm_inf)
        end    
        err = norm(psi-psi_ref)
        if (row==1) 
            @printf("%3i%12.3e%12.3e\n", row, Float64(dt1), Float64(err))
            tab[row,1] = dt1
            tab[row,2] = err
            tab[row,3] = 0 
        else
            p = log(err_old/err)/log(2.0);
            @printf("%3i%12.3e%12.3e%7.2f\n", row, Float64(dt1), Float64(err), Float64(p))
            tab[row,1] = dt1
            tab[row,2] = err
            tab[row,3] = p 
        end
        err_old = err
        dt1 = 0.5*dt1
        psi = copy(wf_save_initial_value)
    end
    tab
end





function Gamma4!(h::HubbardHamiltonian,
                 r::Vector{Complex{Float64}}, u::Vector{Complex{Float64}}, 
                 t::Float64, b::Float64, f::Complex{Float64}, fd::Complex{Float64})
    set_fac_diag!(h, b) 
    A = fd
    B = f
    n = size(h, 2)
    s1 = unsafe_wrap(Array, pointer(h.wsp, 1),   n, own=false)
    s2 = unsafe_wrap(Array, pointer(h.wsp, n+1), n, own=false)

    # s1 = B*u
      set_fac_offdiag!(h, B)
      A_mul_B!(s1, h, u)
    # r = c_B*s1, c_B=1 
      r[:] = s1[:] # copy
    # s2 = A*u
      set_fac_offdiag!(h, A)
      A_mul_B!(s2, h, u)
    # r += c_A*s2, c_A=t 
      BLAS.axpy!(t, s2, r)
    # s2 = B*s2
      set_fac_offdiag!(h, B)
      A_mul_B!(s2, h, s2)
    # r += c_BA*s2, c_BA=-1/2*t^2 
      BLAS.axpy!(-t^2/2, s2, r)
    # s2 = B*s2
      set_fac_offdiag!(h, B)
      A_mul_B!(s2, h, s2)
    # r += c_BBA*s2, c_BBA=1/6*t^3
      BLAS.axpy!(t^3/6, s2, r)
    # s2 = B*s2
      set_fac_offdiag!(h, B)
      A_mul_B!(s2, h, s2)
    # r += c_BBBA*s2, c_BBBA=1/24*t^4
      BLAS.axpy!(t^4/24, s2, r)

    # s2 = A*s1
      set_fac_offdiag!(h, A)
      A_mul_B!(s2, h, s1)
    # r += c_AB*s2, c_AB=1/2*t^2
      BLAS.axpy!(t^2/2, s2, r)
    # s2 = B*s2
      set_fac_offdiag!(h, B)
      A_mul_B!(s2, h, s2)
    # r += c_BAB*s2, c_BAB=-1/3*t^3
      BLAS.axpy!(-t^3/3, s2, r)
    # s2 = B*s2
      set_fac_offdiag!(h, B)
      A_mul_B!(s2, h, s2)
    # r += c_BBAB*s2, c_BBAB=-1/8*t^4
      BLAS.axpy!(-t^4/8, s2, r)

    # s2 = B*s1
      set_fac_offdiag!(h, B)
      A_mul_B!(s2, h, s1)
    # s1 = A*s2
      set_fac_offdiag!(h, A)
      A_mul_B!(s1, h, s2)
    # r += c_ABB*s1, c_ABB=1/6t^3 
      BLAS.axpy!(t^3/6, s1, r)
    # s1 = B*s1
      set_fac_offdiag!(h, B)
      A_mul_B!(s1, h, s1)
    # r += c_BABB*s1, c_BABB=1/8t^4
      BLAS.axpy!(t^4/8, s1, r)

    # s1 = B*s2
      set_fac_offdiag!(h, B)
      A_mul_B!(s1, h, s2)
    # s1 = A*s1
      set_fac_offdiag!(h, A)
      A_mul_B!(s1, h, s1)
    # r += c_ABBB*s1, c_ABBB=-1/24*t^4 
      BLAS.axpy!(-t^4/24, s1, r)

end
