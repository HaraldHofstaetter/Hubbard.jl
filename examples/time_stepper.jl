using FExpokit

import FExpokit: get_lwsp_liwsp_expv 

get_lwsp_liwsp_expv(H, scheme, m::Integer=30) = get_lwsp_liwsp_expv(size(H, 2), m)

struct CommutatorFreeScheme
    A::Array{Float64,2}
    c::Array{Float64,1}
end

CF2 = CommutatorFreeScheme( ones(1,1), [1/2] )

CF4 = CommutatorFreeScheme(
    [1/4+sqrt(3)/6 1/4-sqrt(3)/6
     1/4-sqrt(3)/6 1/4+sqrt(3)/6],
    [1/2-sqrt(3)/6, 1/2+sqrt(3)/6])



function step!(psi::Array{Complex{Float64},1}, H, f::Function, 
               t::Real, dt::Real, scheme::CommutatorFreeScheme)
    state = save_state(H)

    set_matrix_times_minus_i!(H, false) # this is done by expv
    for j=1:size(scheme.A, 1)
        set_fac!(H, sum(scheme.A[j,:]), sum(scheme.A[j,:].*f.(t+dt*scheme.c)))

        expv!(psi, dt, H, psi, anorm=get_norm0(H), 
              matrix_times_minus_i=true, hermitian=true,
              wsp=get_wsp(H), iwsp=get_iwsp(H))

    end

    restore_state!(H, state)
end    

abstract type Magnus2 end

struct Magnus2Exponent
    H
    f::Complex{Float64}
end

import Base.LinAlg: size 
size(E::Magnus2Exponent, dim) = size(E.H, dim)

import Base.LinAlg: A_mul_B!

function A_mul_B!(w::Vector{Complex{Float64}}, E::Magnus2Exponent, 
                  v::Vector{Complex{Float64}})
    set_matrix_times_minus_i!(H, false)
    set_fac!(E.H, 1.0, E.f)
    A_mul_B!(w, E.H, v)
end

function step!(psi::Array{Complex{Float64},1}, H, f::Function, 
               t::Real, dt::Real, scheme::Type{Magnus2})
    state = save_state(H)

    E = Magnus2Exponent(H, f(t+0.5*dt))
    expv!(psi, dt, E, psi, anorm=get_norm0(H), 
          matrix_times_minus_i=true, hermitian=true,
          wsp=get_wsp(H), iwsp=get_iwsp(H))

    restore_state!(H, state)
end   


abstract type Magnus4 end

function get_lwsp_liwsp_expv(H, scheme::Type{Magnus4}, m::Integer=30) 
    (lw, liw) = get_lwsp_liwsp_expv(size(H, 2), m)
    (lw+size(H, 2), liw)
end

struct Magnus4Exponent
    H
    c::Complex{Float64}
    f1::Complex{Float64}
    f2::Complex{Float64}
end

import Base.LinAlg: size 
size(E::Magnus4Exponent, dim) = size(E.H, dim)

import Base.LinAlg: A_mul_B!

function A_mul_B!(w::Vector{Complex{Float64}}, E::Magnus4Exponent, 
                  v::Vector{Complex{Float64}})
    set_matrix_times_minus_i!(E.H, false)
    n = size(E.H, 2)
    s = unsafe_wrap(Array, pointer(get_wsp(E.H), length(get_wsp(E.H))-n+1), n, false)

    set_fac!(E.H, 1.0, E.f1)
    A_mul_B!(s, E.H, v)

    w[:] = 0.5*s[:]
    
    set_fac!(E.H, 1.0, E.f2)
    A_mul_B!(s, E.H, s)
    
    w[:] -= E.c*s

    set_fac!(E.H, 1.0, E.f2)
    A_mul_B!(s, E.H, v)

    w[:] += 0.5*s

    set_fac!(E.H, 1.0, E.f1)
    A_mul_B!(s, E.H, s)
    
    w[:] += E.c*s
end


function step!(psi::Array{Complex{Float64},1}, H, f::Function, 
               t::Real, dt::Real, scheme::Type{Magnus4})
    state = save_state(H)

    E = Magnus4Exponent(H, dt*sqrt(3)/12*1im, 
                           f(t+dt*(1/2-sqrt(3)/6)), 
                           f(t+dt*(1/2+sqrt(3)/6)))
    expv!(psi, dt, E, psi, anorm=get_norm0(H), 
          matrix_times_minus_i=true, hermitian=true,
          wsp=get_wsp(H), iwsp=get_iwsp(H))

    restore_state!(H, state)
end   


struct EquidistantTimeStepper
    H
    f::Function
    psi::Array{Complex{Float64},1}
    t0::Float64
    tend::Float64
    dt::Float64
    scheme
    function EquidistantTimeStepper(H, f::Function, 
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real; scheme=CF4)

        # allocate workspace
        lwsp, liwsp = get_lwsp_liwsp_expv(H, scheme)  
        set_wsp!(H, lwsp)
        set_iwsp!(H, liwsp)
        
        new(H, f, psi, t0, tend, dt, scheme)
    end
end

Base.start(ets::EquidistantTimeStepper) = ets.t0

function Base.done(ets::EquidistantTimeStepper, t) 
    if (t >= ets.tend)
        # deallocate workspace
        set_wsp!(ets.H, 0)
        set_iwsp!(ets.H, 0)
        return true
    end
    false
end

function Base.next(ets::EquidistantTimeStepper, t)
    step!(ets.psi, ets.H, ets.f, t, ets.dt, ets.scheme)
    t1 = t + ets.dt < ets.tend ? t + ets.dt : ets.tend
    t1, t1
end

function local_orders(H, f::Function,
                      psi::Array{Complex{Float64},1}, t0::Real, dt::Real; 
                      scheme=CF2, reference_scheme=scheme, 
                      reference_steps=10,
                      rows=8)
    tab = zeros(Float64, rows, 3)

    # allocate workspace
    lwsp, liwsp = get_lwsp_liwsp_expv(H, scheme)  
    set_wsp!(H, lwsp)
    set_iwsp!(H, liwsp)

    wf_save_initial_value = copy(psi)
    psi_ref = copy(psi)

    dt1 = dt
    err_old = 0.0
    println("             dt         err      p")
    println("-----------------------------------")
    for row=1:rows
        step!(psi, H, f, t0, dt1, scheme)
        psi_ref = copy(wf_save_initial_value)
        dt1_ref = dt1/reference_steps
        for k=1:reference_steps
            step!(psi_ref, H, f, t0+(k-1)*dt1_ref, dt1_ref, reference_scheme)
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
    set_wsp!(H, 0)
    set_iwsp!(H, 0)

    tab
end

function local_orders_est(H, f::Function, fd::Function,
                      psi::Array{Complex{Float64},1}, t0::Real, dt::Real; 
                      scheme=CF2_defectbased, reference_scheme=CF4, 
                      reference_steps=10,
                      rows=8)
    tab = zeros(Float64, rows, 5)

    # allocate workspace
    lwsp, liwsp = get_lwsp_liwsp_expv(H, scheme)  
    set_wsp!(H, lwsp)
    set_iwsp!(H, liwsp)

    wf_save_initial_value = copy(psi)
    psi_ref = copy(psi)
    psi_est = copy(psi)

    dt1 = dt
    err_old = 0.0
    err_est_old = 0.0
    println("             dt         err      p       err_est      p")
    println("--------------------------------------------------------")
    for row=1:rows
        step_estimated!(psi, psi_est, H, f, fd, t0, dt1, scheme)
        psi_ref = copy(wf_save_initial_value)
        dt1_ref = dt1/reference_steps
        for k=1:reference_steps
            step!(psi_ref, H, f, t0+(k-1)*dt1_ref, dt1_ref, reference_scheme)
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
    set_wsp!(H, 0)
    set_iwsp!(H, 0)

    tab
end


function Gamma2!(r::Vector{Complex{Float64}}, H,
                 u::Vector{Complex{Float64}}, dt::Float64, 
                 f::Complex{Float64}, fd::Complex{Float64},
                 s1::Vector{Complex{Float64}}, s2::Vector{Complex{Float64}})
    A = fd
    B = f

    # s1 = B*u
      set_fac!(H, 1.0, B)
      A_mul_B!(s1, H, u)
    # r = c_B*s1, c_B=1 
      r[:] = s1[:] # copy
    # s2 = A*u
      set_fac!(H, 0.0, A)
      A_mul_B!(s2, H, u)
    # r += c_A*s2, c_A=dt 
      r[:] += dt*s2
    # s2 = B*s2
      set_fac!(H, 1.0, B)
      A_mul_B!(s2, H, s2)
    # r += c_BA*s2, c_BA=1/2*dt^2 
      r[:] += (dt^2/2)*s2

    # s2 = A*s1
      set_fac!(H, 0.0, A)
      A_mul_B!(s2, H, s1)
    # r += c_AB*s2, c_AB=-1/2*dt^2
      r[:] -= (dt^2/2)*s2
end

abstract type CF2_defectbased end

get_order(::Type{CF2_defectbased}) = 2

function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H, f::Function, fd::Function, t::Real, dt::Real,
                 scheme::Type{CF2_defectbased})
    state = save_state(H)

    n = size(H, 2)
    s1 = unsafe_wrap(Array, pointer(get_wsp(H), 1),   n, false)
    s2 = unsafe_wrap(Array, pointer(get_wsp(H), n+1), n, false)

    # psi = S(dt)*psi
    set_fac!(H, 1.0, f(t+0.5*dt))
    set_matrix_times_minus_i!(H, false) # this is done by expv
    expv!(psi, dt, H, psi, anorm=get_norm0(H), 
          matrix_times_minus_i=true, hermitian=true,
          wsp=get_wsp(H), iwsp=get_iwsp(H))
    set_matrix_times_minus_i!(H, true)

    # psi_est = Gamma(dt)*psi-A(t+dt)*psi
    #   psi_est = Gamma(dt)*psi
    Gamma2!(psi_est, H, psi, dt, f(t+0.5*dt), 0.5*fd(t+0.5*dt), s1, s2)
    #   s1 = A(t+dt)*psi
    set_fac!(H, 1.0, f(t+dt))
    LinAlg.A_mul_B!(s1, H, psi)
    #   psi_est = psi_est - s1
    psi_est[:] -= s1[:]

    # psi_est = psi_est*(dt/3)
    psi_est[:] *= dt/3

    restore_state!(H, state)
end

function Gamma4!(r::Vector{Complex{Float64}}, H,
                 u::Vector{Complex{Float64}}, dt::Float64, 
                 g::Float64, f::Complex{Float64}, fd::Complex{Float64},
                 s1::Vector{Complex{Float64}}, s2::Vector{Complex{Float64}})
    A = fd
    B = f

    # s1 = B*u
      set_fac!(H, g, B)
      A_mul_B!(s1, H, u)
    # r = c_B*s1, c_B=1 
      r[:] = s1[:] # copy
    # s2 = A*u
      set_fac!(H, 0.0, A)
      A_mul_B!(s2, H, u)
    # r += c_A*s2, c_A=dt 
      r[:] += dt*s2
    # s2 = B*s2
      set_fac!(H, g, B)
      A_mul_B!(s2, H, s2)
    # r += c_BA*s2, c_BA=1/2*dt^2 
      r[:] += (dt^2/2)*s2
    # s2 = B*s2
      set_fac!(H, g, B)
      A_mul_B!(s2, H, s2)
    # r += c_BBA*s2, c_BBA=1/6*dt^3
      r[:] += (dt^3/6)*s2
    # s2 = B*s2
      set_fac!(H, g, B)
      A_mul_B!(s2, H, s2)
    # r += c_BBBA*s2, c_BBBA=1/24*dt^4
      r[:] += (dt^4/24)*s2

    # s2 = A*s1
      set_fac!(H, 0.0, A)
      A_mul_B!(s2, H, s1)
    # r += c_AB*s2, c_AB=-1/2*dt^2
      r[:] -= (dt^2/2)*s2
    # s2 = B*s2
      set_fac!(H, g, B)
      A_mul_B!(s2, H, s2)
    # r += c_BAB*s2, c_BAB=-1/3*dt^3
      r[:] -= (dt^3/3)*s2
    # s2 = B*s2
      set_fac!(H, g, B)
      A_mul_B!(s2, H, s2)
    # r += c_BBAB*s2, c_BBAB=-1/8*dt^4
      r[:] -= (dt^4/8)*s2

    # s2 = B*s1
      set_fac!(H, g, B)
      A_mul_B!(s2, H, s1)
    # s1 = A*s2
      set_fac!(H, 0.0, A)
      A_mul_B!(s1, H, s2)
    # r += c_ABB*s1, c_ABB=1/6dt^3 
      r[:] += (dt^3/6)*s1
    # s1 = B*s1
      set_fac!(H, g, B)
      A_mul_B!(s1, H, s1)
    # r += c_BABB*s1, c_BABB=1/8dt^4
      r[:] += (dt^4/8)*s1

    # s1 = B*s2
      set_fac!(H, g, B)
      A_mul_B!(s1, H, s2)
    # s1 = A*s1
      set_fac!(H, 0.0, A)
      A_mul_B!(s1, H, s1)
    # r += c_ABBB*s1, c_ABBB=-1/24*dt^4 
      r[:] -= (dt^4/24)*s1
end


abstract type CF4_defectbased end

get_order(::Type{CF4_defectbased}) = 4

function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H, f::Function, fd::Function, t::Real, dt::Real,
                 scheme::Type{CF4_defectbased})
    state = save_state(H)

    n = size(H, 2)
    s = unsafe_wrap(Array, pointer(get_wsp(H), 1), n, false)
    s1 = unsafe_wrap(Array, pointer(get_wsp(H), n+1),   n, false)
    s2 = unsafe_wrap(Array, pointer(get_wsp(H), 2*n+1), n, false)

    CF4 = CommutatorFreeScheme(
    [1/4+sqrt(3)/6 1/4-sqrt(3)/6
     1/4-sqrt(3)/6 1/4+sqrt(3)/6],
    [1/2-sqrt(3)/6, 1/2+sqrt(3)/6])     
    
    # psi = S_1(dt)*psi
    set_fac!(H, sum(CF4.A[1,:]), sum(CF4.A[1,:].*f.(t+dt*CF4.c)))
    set_matrix_times_minus_i!(H, false) # this is done by expv
    expv!(psi, dt, H, psi, anorm=get_norm0(H), 
          matrix_times_minus_i=true, hermitian=true,
          wsp=get_wsp(H), iwsp=get_iwsp(H))
    set_matrix_times_minus_i!(H, true)

    # psi_est = Gamma_1(dt)*psi
    Gamma4!(psi_est, H, psi, dt, 
            sum(CF4.A[1,:]),
            sum(CF4.A[1,:].*f.(t+dt*CF4.c)), 
            sum(CF4.c.*CF4.A[1,:].*fd.(t+dt*CF4.c)),
            s1, s2)

    # psi_est = S_2(dt)*psi_est
    set_fac!(H, sum(CF4.A[2,:]), sum(CF4.A[2,:].*f.(t+dt*CF4.c)))
    set_matrix_times_minus_i!(H, false) # this is done by expv
    expv!(psi_est, dt, H, psi_est, anorm=get_norm0(H), 
          matrix_times_minus_i=true, hermitian=true,
          wsp=get_wsp(H), iwsp=get_iwsp(H)) 
    set_matrix_times_minus_i!(H, true)

    # psi = S_2(dt)*psi
    set_fac!(H, sum(CF4.A[2,:]), sum(CF4.A[2,:].*f.(t+dt*CF4.c)))
    set_matrix_times_minus_i!(H, false) # this is done by expv
    expv!(psi, dt, H, psi, anorm=get_norm0(H), 
          matrix_times_minus_i=true, hermitian=true,
          wsp=get_wsp(H), iwsp=get_iwsp(H)) 
    set_matrix_times_minus_i!(H, true)

    # psi_est = psi_est+Gamma_2(dt)*psi-A(t+dt)*psi
    #  s = Gamma_2(dt)*psi
    Gamma4!(s, H, psi, dt, 
            sum(CF4.A[2,:]),
            sum(CF4.A[2,:].*f.(t+dt*CF4.c)), 
            sum(CF4.c.*CF4.A[2,:].*fd.(t+dt*CF4.c)),
            s1, s2)

    # psi_est = psi_est+s
    psi_est[:] += s[:]
    #  s = A(t+dt)*psi
    set_fac!(H, 1.0, f(t+dt))
    LinAlg.A_mul_B!(s, H, psi)
    #  psi_est = psi_est-s
    psi_est[:] -= s[:]

    # psi_est = psi_est*(dt/5)
    psi_est[:] *= dt/5.0

    restore_state!(H, state)
end


abstract type Magnus4_defectbased end

get_order(::Type{Magnus4_defectbased}) = 4

function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H, f::Function, fd::Function, t::Real, dt::Real,
                 scheme::Type{Magnus4_defectbased})
    state = save_state(H)

    n = size(H, 2)
    s = unsafe_wrap(Array, pointer(get_wsp(H), 1), n, false)

    step!(psi, H, f, t0, dt, Magnus4)
   
    ### TODO...

    restore_state!(H, state)
end




struct AdaptiveTimeStepper
    H
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

    function AdaptiveTimeStepper(H, 
                 f::Function, fd::Function,
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real,  tol::Real; scheme=CF2_defectbased)
        order = get_order(scheme)

        # allocate workspace
        lwsp, liwsp = get_lwsp_liwsp_expv(H, scheme)  
        set_wsp!(H, lwsp)
        set_iwsp!(H, liwsp)

        psi_est = zeros(Complex{Float64}, size(H, 2))
        psi0 = zeros(Complex{Float64}, size(H, 2))
        
        new(H, f, fd, psi, t0, tend, dt, tol, order, scheme, psi_est, psi0)
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
        step_estimated!(ats.psi, ats.psi_est, ats.H, ats.f, ats.fd, ats.t0, dt, ats.scheme)
        err = norm(ats.psi_est)/ats.tol
        dt = dt*min(facmax, max(facmin, fac*(1.0/err)^(1.0/(ats.order+1))))
        if err>=1.0
           copy!(ats.psi, ats.psi0)
           @printf("t=%17.9e  err=%17.8e  dt=%17.8e  rejected...\n", Float64(state.t), Float64(err), Float64(dt))
        end   
    end
    state.t + dt0, AdaptiveTimeStepperState(state.t+dt0, dt)
end
