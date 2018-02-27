using FExpokit

import FExpokit: get_lwsp_liwsp_expv 

get_lwsp_liwsp_expv(H, scheme, m::Integer=30) = get_lwsp_liwsp_expv(size(H, 2), m)


struct CommutatorFreeScheme
    A::Array{Float64,2}
    c::Array{Float64,1}
    p::Int
end

get_order(scheme::CommutatorFreeScheme) = scheme.p

CF2 = CommutatorFreeScheme( ones(1,1), [1/2], 2 )

CF4 = CommutatorFreeScheme(
    [1/4+sqrt(3)/6 1/4-sqrt(3)/6
     1/4-sqrt(3)/6 1/4+sqrt(3)/6],
    [1/2-sqrt(3)/6, 1/2+sqrt(3)/6],
     4)

CF4o = CommutatorFreeScheme(
    [37/240+10/87*sqrt(5/3) -1/30  37/240-10/87*sqrt(5/3)
     -11/360                23/45  -11/360
     37/240-10/87*sqrt(5/3) -1/30  37/240+10/87*sqrt(5/3)],
     [1/2-sqrt(15)/10, 1/2, 1/2+sqrt(15)/10],
     4)

CF6 = CommutatorFreeScheme(
  [ 0.2158389969757678 -0.0767179645915514  0.0208789676157837
   -0.0808977963208530 -0.1787472175371576  0.0322633664310473 
    0.1806284600558301  0.4776874043509313 -0.0909342169797981
   -0.0909342169797981  0.4776874043509313  0.1806284600558301
    0.0322633664310473 -0.1787472175371576 -0.0808977963208530 
    0.0208789676157837 -0.0767179645915514  0.2158389969757678],
  [1/2-sqrt(15)/10, 1/2, 1/2+sqrt(15)/10],
  6)


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


function Gamma!(r::Vector{Complex{Float64}}, H,
                u::Vector{Complex{Float64}}, p::Int, dt::Float64, 
                g::Float64, f::Complex{Float64}, fd::Complex{Float64},
                s1::Vector{Complex{Float64}}, s2::Vector{Complex{Float64}})
    
    #s2=B*u
    set_fac!(H, g, f);  A_mul_B!(s2, H, u)
    r[:] = s2[:]    
    
    if p>=1
        #s1=A*u
        set_fac!(H, 0.0, fd);  A_mul_B!(s1, H, u)
        r[:] += dt*s1[:] 
    end
    if p>=2
        #s1=B*s1=BAu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] += (dt^2/2)*s1[:] 
    end
    if p>=3
        #s1=B*s1=BBAu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] += (dt^3/6)*s1[:] 
    end
    if p>=4
        #s1=B*s1=BBBAu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] += (dt^4/24)*s1[:] 
    end
    if p>=5
        #s1=B*s1=BBBBAu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] += (dt^5/120)*s1[:] 
    end
    if p>=6
        #s1=B*s1=BBBBBAu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] += (dt^6/720)*s1[:] 
    end

    if p>=2
        #s1=A*s2=ABu
        set_fac!(H, 0.0, fd);  A_mul_B!(s1, H, s2)
        r[:] -= (dt^2/2)*s1[:] 
    end
    if p>=3
        #s1=B*s1=BABu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] -= (dt^3/3)*s1[:] 
    end
    if p>=4
        #s1=B*s1=BBABu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] -= (dt^4/8)*s1[:] 
    end
    if p>=5
        #s1=B*s1=BBBABu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] -= (dt^5/30)*s1[:] 
    end
    if p>=6
        #s1=B*s1=BBBBABu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] -= (dt^6/144)*s1[:] 
    end

    if p>=3
        #s2=B*s2=BBu
        set_fac!(H, g, f);  A_mul_B!(s2, H, s2)
        #s1=A*s2=ABBu
        set_fac!(H, 0.0, fd);  A_mul_B!(s1, H, s2)
        r[:] += (dt^3/6)*s1
    end
    if p>=4
        #s1=B*s1=BABBu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] += (dt^4/8)*s1
    end
    if p>=5
        #s1=B*s1=BBABBu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] += (dt^5/20)*s1
    end
    if p>=6
        #s1=B*s1=BBBABBu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] += (dt^6/72)*s1
    end

    if p>=4
        #s2=B*s2=BBBu
        set_fac!(H, g, f);  A_mul_B!(s2, H, s2)
        #s1=A*s2=ABBBu
        set_fac!(H, 0.0, fd);  A_mul_B!(s1, H, s2)
        r[:] -= (dt^4/24)*s1
    end
    if p>=5
        #s1=B*s1=BABBBu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] -= (dt^5/30)*s1
    end
    if p>=6
        #s1=B*s1=BBABBBu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] -= (dt^6/72)*s1
    end

    if p>=5
        #s2=B*s2=BBBBu
        set_fac!(H, g, f);  A_mul_B!(s2, H, s2)
        #s1=A*s2=ABBBBu
        set_fac!(H, 0.0, fd);  A_mul_B!(s1, H, s2)
        r[:] += (dt^5/120)*s1
    end
    if p>=6
        #s1=B*s1=BABBBBu
        set_fac!(H, g, f);  A_mul_B!(s1, H, s1)
        r[:] += (dt^6/144)*s1
    end

    if p>=6
        #s2=B*s2=BBBBBu
        set_fac!(H, g, f);  A_mul_B!(s2, H, s2)
        #s1=A*s2=ABBBBBu
        set_fac!(H, 0.0, fd);  A_mul_B!(s1, H, s2)
        r[:] -= (dt^6/720)*s1
    end
end

function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H, f::Function, fd::Function, t::Real, dt::Real,
                 ::Type{Magnus2};
                 symmetrized_defect::Bool=false)
    state = save_state(H)

    n = size(H, 2)
    s = unsafe_wrap(Array, pointer(get_wsp(H), 1), n, false)

    set_fac!(H, -0.5, -0.5*f(t))
    set_matrix_times_minus_i!(H, true)
    LinAlg.A_mul_B!(psi_est, H, psi)

    set_fac!(H, 1.0, f(t+0.5*dt))
    set_matrix_times_minus_i!(H, false) # this is done by expv
    expv!(psi_est, dt, H, psi_est, anorm=get_norm0(H), 
          matrix_times_minus_i=true, hermitian=true,
          wsp=get_wsp(H), iwsp=get_iwsp(H))
    set_matrix_times_minus_i!(H, true)
    
    set_fac!(H, 1.0, f(t+0.5*dt))
    set_matrix_times_minus_i!(H, false) # this is done by expv
    expv!(psi, dt, H, psi, anorm=get_norm0(H), 
          matrix_times_minus_i=true, hermitian=true,
          wsp=get_wsp(H), iwsp=get_iwsp(H))
    set_matrix_times_minus_i!(H, true)
    
    set_fac!(H, 1.0, f(t+0.5*dt))
    set_matrix_times_minus_i!(H, true)
    LinAlg.A_mul_B!(s, H, psi)
    psi_est[:] += s[:]

    set_fac!(H, 0.5, 0.5*f(t+dt))
    set_matrix_times_minus_i!(H, true)
    LinAlg.A_mul_B!(s, H, psi)
    psi_est[:] -= s[:]
    
    psi_est[:] *= dt/3
    
    restore_state!(H, state)
end

function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H, f::Function, fd::Function, t::Real, dt::Real,
                 scheme::CommutatorFreeScheme;
                 symmetrized_defect::Bool=false)
    state = save_state(H)

    n = size(H, 2)
    s = unsafe_wrap(Array, pointer(get_wsp(H), 1), n, false)
    s1 = unsafe_wrap(Array, pointer(get_wsp(H), n+1),   n, false)
    s2 = unsafe_wrap(Array, pointer(get_wsp(H), 2*n+1), n, false)

    if symmetrized_defect
        set_fac!(H, -0.5, -0.5*f(t))
        set_matrix_times_minus_i!(H, true) 
        LinAlg.A_mul_B!(psi_est, H, psi)

        set_fac!(H, sum(scheme.A[1,:]), sum(scheme.A[1,:].*f.(t+dt*scheme.c)))
        set_matrix_times_minus_i!(H, false) # this is done by expv
        expv!(psi_est, dt, H, psi_est, anorm=get_norm0(H), 
              matrix_times_minus_i=true, hermitian=true,
              wsp=get_wsp(H), iwsp=get_iwsp(H)) 
        set_matrix_times_minus_i!(H, true)
    else
        psi_est[:] = 0.0
    end

    set_fac!(H, sum(scheme.A[1,:]), sum(scheme.A[1,:].*f.(t+dt*scheme.c)))
    set_matrix_times_minus_i!(H, false) # this is done by expv
    expv!(psi, dt, H, psi, anorm=get_norm0(H), 
          matrix_times_minus_i=true, hermitian=true,
          wsp=get_wsp(H), iwsp=get_iwsp(H))
    set_matrix_times_minus_i!(H, true)

    Gamma!(s, H, psi, scheme.p, dt,
            sum(scheme.A[1,:]),
            sum(scheme.A[1,:].*f.(t+dt*scheme.c)), 
            symmetrized_defect?
               sum((scheme.c-0.5).*scheme.A[1,:].*fd.(t+dt*scheme.c)):
               sum(scheme.c.*scheme.A[1,:].*fd.(t+dt*scheme.c)),
            s1, s2)
    psi_est[:] += s[:]

    for j=2:size(scheme.A, 1)

        set_fac!(H, sum(scheme.A[j,:]), sum(scheme.A[j,:].*f.(t+dt*scheme.c)))
        set_matrix_times_minus_i!(H, false) # this is done by expv
        expv!(psi_est, dt, H, psi_est, anorm=get_norm0(H), 
              matrix_times_minus_i=true, hermitian=true,
              wsp=get_wsp(H), iwsp=get_iwsp(H)) 
        set_matrix_times_minus_i!(H, true)

        set_fac!(H, sum(scheme.A[j,:]), sum(scheme.A[j,:].*f.(t+dt*scheme.c)))
        set_matrix_times_minus_i!(H, false) # this is done by expv
        expv!(psi, dt, H, psi, anorm=get_norm0(H), 
              matrix_times_minus_i=true, hermitian=true,
              wsp=get_wsp(H), iwsp=get_iwsp(H)) 
        set_matrix_times_minus_i!(H, true)
    
        Gamma!(s, H, psi, scheme.p, dt, 
                sum(scheme.A[j,:]),
                sum(scheme.A[j,:].*f.(t+dt*scheme.c)), 
                symmetrized_defect?
                   sum((scheme.c-0.5).*scheme.A[j,:].*fd.(t+dt*scheme.c)):
                   sum(scheme.c.*scheme.A[j,:].*fd.(t+dt*scheme.c)),
                s1, s2)

        psi_est[:] += s[:]

    end
   
    if symmetrized_defect
        set_fac!(H, 0.5, 0.5*f(t+dt))
    else
        set_fac!(H, 1.0, f(t+dt))
    end
    LinAlg.A_mul_B!(s, H, psi)
    psi_est[:] -= s[:]

    psi_est[:] *= dt/(scheme.p+1)

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



get_order(::Type{Magnus4}) = 4

function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H, f::Function, fd::Function, t::Real, dt::Real,
                 scheme::Type{Magnus4})
    state = save_state(H)

    n = size(H, 2)
    s = unsafe_wrap(Array, pointer(get_wsp(H), 1), n, false)

    step!(psi, H, f, t0, dt, Magnus4)
   
    ### TODO...

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
    if t >= ets.tend
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
                      symmetrized_defect::Bool=false,
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
        step_estimated!(psi, psi_est, H, f, fd, t0, dt1, scheme,
                        symmetrized_defect=symmetrized_defect)
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
                 t0::Real, tend::Real, dt::Real,  tol::Real; scheme=CF4)
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
    if state.t >= ats.tend
        # deallocate workspace
        set_wsp!(ats.H, 0)
        set_iwsp!(ats.H, 0)
        return true
    end
    false
end  

function Base.next(ats::AdaptiveTimeStepper, state::AdaptiveTimeStepperState)
    const facmin = 0.25
    const facmax = 4.0
    const fac = 0.9

    dt = state.dt
    dt0 = dt
    ats.psi0[:] = ats.psi[:]
    err = 2.0
    while err>=1.0
        dt = min(dt, ats.tend-state.t)
        dt0 = dt
        step_estimated!(ats.psi, ats.psi_est, ats.H, ats.f, ats.fd, state.t, dt, ats.scheme)
        err = norm(ats.psi_est)/ats.tol
        dt = dt*min(facmax, max(facmin, fac*(1.0/err)^(1.0/(ats.order+1))))
        if err>=1.0
           ats.psi[:] = ats.psi0
           @printf("t=%17.9e  err=%17.8e  dt=%17.8e  rejected...\n", Float64(state.t), Float64(err), Float64(dt))
        end   
    end
    state.t + dt0, AdaptiveTimeStepperState(state.t+dt0, dt)
end



