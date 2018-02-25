using Hubbard
using FExpokit

import FExpokit: get_lwsp_liwsp_expv 

get_lwsp_liwsp_expv(h::HubbardHamiltonian, scheme, m::Integer=30) = get_lwsp_liwsp_expv(size(h, 2), m)

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
               t::Real, dt::Real, scheme::CommutatorFreeScheme)
    state = save_state(h)

    h.matrix_times_minus_i = false # this is done by expv
    for j=1:size(scheme.A, 1)
        set_fac!(h, sum(scheme.A[j,:]), sum(scheme.A[j,:].*f.(t+dt*scheme.c)))

        expv!(psi, dt, h, psi, anorm=h.norm, 
              matrix_times_minus_i=true, hermitian=true,
              wsp=h.wsp, iwsp=h.iwsp) 

    end

    restore_state!(h, state)
end    

abstract type Magnus2 end

function Magnus2_exponent(w::Vector{Complex{Float64}}, v::Vector{Complex{Float64}},
                          h::HubbardHamiltonian, f1::Complex{Float64}) 
    h.matrix_times_minus_i = false
    set_fac!(h, 1.0, f1)
    A_mul_B!(w, h, v)
end

function step!(psi::Array{Complex{Float64},1}, h::HubbardHamiltonian, f::Function, 
               t::Real, dt::Real, scheme::Type{Magnus2})
    state = save_state(h)

    f1 = f(t+0.5*dt)
    expv!(psi, dt, Magnus2_exponent, psi, h.norm,
          args=(h, f1),
          matrix_times_minus_i=true, hermitian=true,
          wsp=h.wsp, iwsp=h.iwsp) 

    restore_state!(h, state)
end   


abstract type Magnus4 end

function get_lwsp_liwsp_expv(h::HubbardHamiltonian, scheme::Type{Magnus4}, m::Integer=30) 
    (lw, liw) = get_lwsp_liwsp_expv(size(h, 2), m)
    (lw+size(h, 2), liw)
end


function Magnus4_exponent(w::Vector{Complex{Float64}}, v::Vector{Complex{Float64}},
                  h::HubbardHamiltonian, c::Complex{Float64}, f1::Complex{Float64}, f2::Complex{Float64}) 
    h.matrix_times_minus_i = false
    n = size(h, 2)
    s = unsafe_wrap(Array, pointer(h.wsp, length(h.wsp)-n+1), n, false)

    set_fac!(h, 1.0, f1)
    A_mul_B!(s, h, v)

    w[:] = 0.5*s[:]
    
    set_fac!(h, 1.0, f2)
    A_mul_B!(s, h, s)
    
    w[:] -= c*s

    set_fac!(h, 1.0, f2)
    A_mul_B!(s, h, v)

    w[:] += 0.5*s

    set_fac!(h, 1.0, f1)
    A_mul_B!(s, h, s)
    
    w[:] += c*s
end


function step!(psi::Array{Complex{Float64},1}, h::HubbardHamiltonian, f::Function, 
               t::Real, dt::Real, scheme::Type{Magnus4})
    state = save_state(h)
    c = dt*sqrt(3)/12*1im
    f1 = f(t+dt*(1/2-sqrt(3)/6))
    f2 = f(t+dt*(1/2+sqrt(3)/6))
    expv!(psi, dt, Magnus4_exponent, psi, h.norm,
          args=(h, c, f1, f2),
          matrix_times_minus_i=true, hermitian=true,
          wsp=h.wsp, iwsp=h.iwsp) 

    restore_state!(h, state)
end   

struct EquidistantTimeStepper
    h::HubbardHamiltonian
    f::Function
    psi::Array{Complex{Float64},1}
    t0::Float64
    tend::Float
    dt::Float64
    scheme
    function EquidistantTimeStepper(h::HubbardHamiltonian, f::Function, 
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real; scheme=CF4)

        # allocate workspace
        lwsp, liwsp = get_lwsp_liwsp_expv(h, scheme)  
        h.wsp = zeros(Complex{Float64}, lwsp)
        h.iwsp = zeros(Int32, liwsp)
        
        new(h, f, psi, t0, tend, dt, scheme)
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
    step!(ets.psi, ets.h, ets.f, t, ets.dt, ets.scheme)
    t1 = t + ets.dt < ets.tend ? t + ets.dt : ets.tend
    t1, t1
end

function local_orders(h::HubbardHamiltonian, f::Function,
                      psi::Array{Complex{Float64},1}, t0::Real, dt::Real; 
                      scheme=CF2, reference_scheme=scheme, 
                      reference_steps=10,
                      rows=8)
    tab = zeros(Float64, rows, 3)

    # allocate workspace
    lwsp, liwsp = get_lwsp_liwsp_expv(h, scheme)  
    h.wsp = zeros(Complex{Float64}, lwsp)
    h.iwsp = zeros(Int32, liwsp)

    wf_save_initial_value = copy(psi)
    psi_ref = copy(psi)

    dt1 = dt
    err_old = 0.0
    println("             dt         err      p")
    println("-----------------------------------")
    for row=1:rows
        step!(psi, h, f, t0, dt1, scheme)
        psi_ref = copy(wf_save_initial_value)
        dt1_ref = dt1/reference_steps
        for k=1:reference_steps
            step!(psi_ref, h, f, t0+(k-1)*dt1_ref, dt1_ref, reference_scheme)
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
    # deallocate workspace
    h.wsp = Complex{Float64}[]        
    h.iwsp =  Int32[]

    tab
end

function local_orders_est(h::HubbardHamiltonian, f::Function, fd::Function,
                      psi::Array{Complex{Float64},1}, t0::Real, dt::Real; 
                      scheme=CF2_defectbased, reference_scheme=CF4, 
                      reference_steps=10,
                      rows=8)
    tab = zeros(Float64, rows, 5)

    # allocate workspace
    lwsp, liwsp = get_lwsp_liwsp_expv(h, scheme)  
    h.wsp = zeros(Complex{Float64}, lwsp)
    h.iwsp = zeros(Int32, liwsp)

    wf_save_initial_value = copy(psi)
    psi_ref = copy(psi)
    psi_est = copy(psi)

    dt1 = dt
    err_old = 0.0
    err_est_old = 0.0
    println("             dt         err      p       err_est      p")
    println("--------------------------------------------------------")
    for row=1:rows
        step_estimated!(psi, psi_est, h, f, fd, t0, dt1, scheme)
        psi_ref = copy(wf_save_initial_value)
        dt1_ref = dt1/reference_steps
        for k=1:reference_steps
            step!(psi_ref, h, f, t0+(k-1)*dt1_ref, dt1_ref, reference_scheme)
        end    
        err = norm(psi-psi_ref)
        err_est = norm(psi-psi_ref-psi_est)
        if (row==1) 
            @printf("%3i%12.3e%12.3e  %19.3e\n", row, Float64(dt1), Float64(err), Float64(err_est))
            tab[row,1] = dt1
            tab[row,2] = err
            tab[row,3] = 0 
            tab[row,4] = err_est
            tab[row,5] = 0 
        else
            p = log(err_old/err)/log(2.0);
            p_est = log(err_est_old/err_est)/log(2.0);
            @printf("%3i%12.3e%12.3e%7.2f  %12.3e%7.2f\n", 
                    row, Float64(dt1), Float64(err), Float64(p), 
                                       Float64(err_est), Float64(p_est))
            tab[row,1] = dt1
            tab[row,2] = err
            tab[row,3] = p 
            tab[row,4] = err_est
            tab[row,5] = p_est 
        end
        err_old = err
        err_est_old = err_est
        dt1 = 0.5*dt1
        psi = copy(wf_save_initial_value)
    end
    # deallocate workspace
    h.wsp = Complex{Float64}[]        
    h.iwsp =  Int32[]

    tab
end


function Gamma2!(r::Vector{Complex{Float64}}, h::HubbardHamiltonian,
                 u::Vector{Complex{Float64}}, dt::Float64, 
                 f::Complex{Float64}, fd::Complex{Float64})
    A = fd
    B = f
    n = size(h, 2)
    s1 = unsafe_wrap(Array, pointer(h.wsp, n+1),   n, false)
    s2 = unsafe_wrap(Array, pointer(h.wsp, 2*n+1), n, false)

    # s1 = B*u
      set_fac!(h, 1.0, B)
      A_mul_B!(s1, h, u)
    # r = c_B*s1, c_B=1 
      r[:] = s1[:] # copy
    # s2 = A*u
      set_fac!(h, 0.0, A)
      A_mul_B!(s2, h, u)
    # r += c_A*s2, c_A=dt 
      r[:] += dt*s2
    # s2 = B*s2
      set_fac!(h, 1.0, B)
      A_mul_B!(s2, h, s2)
    # r += c_BA*s2, c_BA=1/2*dt^2 
      r[:] += (dt^2/2)*s2

    # s2 = A*s1
      set_fac!(h, 0.0, A)
      A_mul_B!(s2, h, s1)
    # r += c_AB*s2, c_AB=-1/2*dt^2
      r[:] -= (dt^2/2)*s2
end

abstract type CF2_defectbased end

function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 h::HubbardHamiltonian, f::Function, fd::Function, t::Real, dt::Real,
                 scheme::Type{CF2_defectbased})
    state = save_state(h)

    n = size(h, 2)
    s = unsafe_wrap(Array, pointer(h.wsp, 1), n, false)

    # psi = S(dt)*psi
    set_fac!(h, 1.0, f(t+0.5*dt))
    h.matrix_times_minus_i = false # this is done by expv
    expv!(psi, dt, h, psi, anorm=h.norm, 
          matrix_times_minus_i=true, hermitian=true,
          wsp=h.wsp, iwsp=h.iwsp)
    h.matrix_times_minus_i = true

    # psi_est = Gamma(dt)*psi-A(t+dt)*psi
    #   psi_est = Gamma(dt)*psi
    Gamma2!(psi_est, h, psi, dt, f(t+0.5*dt), 0.5*fd(t+0.5*dt))
    #   s = A(t+dt)*psi
    set_fac!(h, 1.0, f(t+dt))
    LinAlg.A_mul_B!(s, h, psi)
    #   psi_est = psi_est - s
    psi_est[:] -= s[:]

    # psi_est = psi_est*(dt/3)
    psi_est[:] *= dt/3

    restore_state!(h, state)
end

function Gamma4!(r::Vector{Complex{Float64}}, h::HubbardHamiltonian,
                 u::Vector{Complex{Float64}}, dt::Float64, 
                 g::Float64, f::Complex{Float64}, fd::Complex{Float64})
    A = fd
    B = f
    n = size(h, 2)
    s1 = unsafe_wrap(Array, pointer(h.wsp, n+1),   n, false)
    s2 = unsafe_wrap(Array, pointer(h.wsp, 2*n+1), n, false)

    # s1 = B*u
      set_fac!(h, g, B)
      A_mul_B!(s1, h, u)
    # r = c_B*s1, c_B=1 
      r[:] = s1[:] # copy
    # s2 = A*u
      set_fac!(h, 0.0, A)
      A_mul_B!(s2, h, u)
    # r += c_A*s2, c_A=dt 
      r[:] += dt*s2
    # s2 = B*s2
      set_fac!(h, g, B)
      A_mul_B!(s2, h, s2)
    # r += c_BA*s2, c_BA=1/2*dt^2 
      r[:] += (dt^2/2)*s2
    # s2 = B*s2
      set_fac!(h, g, B)
      A_mul_B!(s2, h, s2)
    # r += c_BBA*s2, c_BBA=1/6*dt^3
      r[:] += (dt^3/6)*s2
    # s2 = B*s2
      set_fac!(h, g, B)
      A_mul_B!(s2, h, s2)
    # r += c_BBBA*s2, c_BBBA=1/24*dt^4
      r[:] += (dt^4/24)*s2

    # s2 = A*s1
      set_fac!(h, 0.0, A)
      A_mul_B!(s2, h, s1)
    # r += c_AB*s2, c_AB=-1/2*dt^2
      r[:] -= (dt^2/2)*s2
    # s2 = B*s2
      set_fac!(h, g, B)
      A_mul_B!(s2, h, s2)
    # r += c_BAB*s2, c_BAB=-1/3*dt^3
      r[:] -= (dt^3/3)*s2
    # s2 = B*s2
      set_fac!(h, g, B)
      A_mul_B!(s2, h, s2)
    # r += c_BBAB*s2, c_BBAB=-1/8*dt^4
      r[:] -= (dt^4/8)*s2

    # s2 = B*s1
      set_fac!(h, g, B)
      A_mul_B!(s2, h, s1)
    # s1 = A*s2
      set_fac!(h, 0.0, A)
      A_mul_B!(s1, h, s2)
    # r += c_ABB*s1, c_ABB=1/6dt^3 
      r[:] += (dt^3/6)*s1
    # s1 = B*s1
      set_fac!(h, g, B)
      A_mul_B!(s1, h, s1)
    # r += c_BABB*s1, c_BABB=1/8dt^4
      r[:] += (dt^4/8)*s1

    # s1 = B*s2
      set_fac!(h, g, B)
      A_mul_B!(s1, h, s2)
    # s1 = A*s1
      set_fac!(h, 0.0, A)
      A_mul_B!(s1, h, s1)
    # r += c_ABBB*s1, c_ABBB=-1/24*dt^4 
      r[:] -= (dt^4/24)*s1
end


abstract type CF4_defectbased end

function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 h::HubbardHamiltonian, f::Function, fd::Function, t::Real, dt::Real,
                 scheme::Type{CF4_defectbased})
    state = save_state(h)

    n = size(h, 2)
    s = unsafe_wrap(Array, pointer(h.wsp, 1), n, false)

    CF4 = CommutatorFreeScheme(
    [1/4+sqrt(3)/6 1/4-sqrt(3)/6
     1/4-sqrt(3)/6 1/4+sqrt(3)/6],
    [1/2-sqrt(3)/6, 1/2+sqrt(3)/6])     
    
    # psi = S_1(dt)*psi
    set_fac!(h, sum(CF4.A[1,:]), sum(CF4.A[1,:].*f.(t+dt*CF4.c)))
    h.matrix_times_minus_i = false # this is done by expv
    expv!(psi, dt, h, psi, anorm=h.norm, 
          matrix_times_minus_i=true, hermitian=true,
          wsp=h.wsp, iwsp=h.iwsp)
    h.matrix_times_minus_i = true

    # psi_est = Gamma_1(dt)*psi
    Gamma4!(psi_est, h, psi, dt, 
            sum(CF4.A[1,:]),
            sum(CF4.A[1,:].*f.(t+dt*CF4.c)), 
            sum(CF4.c.*CF4.A[1,:].*fd.(t+dt*CF4.c)))

    # psi_est = S_2(dt)*psi_est
    set_fac!(h, sum(CF4.A[2,:]), sum(CF4.A[2,:].*f.(t+dt*CF4.c)))
    h.matrix_times_minus_i = false # this is done by expv
    expv!(psi_est, dt, h, psi_est, anorm=h.norm, 
          matrix_times_minus_i=true, hermitian=true,
          wsp=h.wsp, iwsp=h.iwsp) 
    h.matrix_times_minus_i = true

    # psi = S_2(dt)*psi
    set_fac!(h, sum(CF4.A[2,:]), sum(CF4.A[2,:].*f.(t+dt*CF4.c)))
    h.matrix_times_minus_i = false # this is done by expv
    expv!(psi, dt, h, psi, anorm=h.norm, 
          matrix_times_minus_i=true, hermitian=true,
          wsp=h.wsp, iwsp=h.iwsp) 
    h.matrix_times_minus_i = true

    # psi_est = psi_est+Gamma_2(dt)*psi-A(t+dt)*psi
    #  s = Gamma_2(dt)*psi
    Gamma4!(s, h, psi, dt, 
            sum(CF4.A[2,:]),
            sum(CF4.A[2,:].*f.(t+dt*CF4.c)), 
            sum(CF4.c.*CF4.A[2,:].*fd.(t+dt*CF4.c)))
    # psi_est = psi_est+s
    psi_est[:] += s[:]
    #  s = A(t+dt)*psi
    set_fac!(h, 1.0, f(t+dt))
    LinAlg.A_mul_B!(s, h, psi)
    #  psi_est = psi_est-s
    psi_est[:] -= s[:]

    # psi_est = psi_est*(dt/5)
    psi_est[:] *= dt/5.0

    restore_state!(h, state)
end

struct AdaptiveTimeStepper
    h::HubbardHamiltonian
    f::Function
    fd::Function
    psi::Array{Complex{Float64},1}
    t0::Float64
    tend::Float64
    dt::Float64
    tol::Float64
    order::Int
    scheme
    psi_est::Array{Complex{Float64},1}
    psi0::Array{Complex{Float64},1}

    function AdaptiveTimeStepper(h::HubbardHamiltonian, 
                 f::Function, fd::Function,
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real tol::Real; scheme=CF2_defectbased)
        order = get_order(scheme)

        # allocate workspace
        lwsp, liwsp = get_lwsp_liwsp_expv(h, scheme)  
        h.wsp = zeros(Complex{Float64}, lwsp)
        h.iwsp = zeros(Int32, liwsp)

        psi_est = zeros(Complex{Float64}, size(h, 2))
        psi0 = zeros(Complex{Float64}, size(h, 2))
        
        new(h, f, fd, psi, t0, tend, dt, tol, order, scheme, psi_est, psi0)
    end
    
end

immutable AdaptiveTimeStepperState
   t::Real
   dt::Real
end   

Base.start(ats::AdaptiveTimeStepper) = AdaptiveTimeStepperState(ats.t0, ats.dt)

function Base.done(ats::AdaptiveTimeStepper, state::AdaptiveTimeStepperState)
  state.t >= ats.tend
end  

function Base.next(ats::AdaptiveTimeStepper, state::AdaptiveTimeStepperState)
    const facmin = 0.25
    const facmax = 4.0
    const fac = 0.9

    dt = state.dt
    dt0 = dt
    copy!(ats.psi0, ats.psi)
    err = 2.0
    while err>=1.0
        dt = min(dt, ats.tend-state.t)
        dt0 = dt
        step_estimated!(ats.psi, ats.psi_est, ats.h, ats.f, ats.fd, ats.t0, dt, ats.scheme)
        err = norm(ats.psi_est)/ats.tol
        dt = dt*min(facmax, max(facmin, fac*(1.0/err)^(1.0/(ats.order+1))))
        if err>=1.0
           copy!(ats.psi, ats.psi0)
           @printf("t=%17.9e  err=%17.8e  dt=%17.8e  rejected...\n", Float64(state.t), Float64(err), Float64(dt))
        end   
    end
    state.t + dt0, AdaptiveTimeStepperState(state.t+dt0, dt)
end
