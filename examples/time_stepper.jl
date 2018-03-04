using FExpokit

import FExpokit: get_lwsp_liwsp_expv 

get_lwsp_liwsp_expv(H, scheme, m::Integer=30) = get_lwsp_liwsp_expv(size(H, 2), m)

const for_expv = 0
const for_Gamma = 1
const for_Gamma_d = 2
const for_Gamma_d_symmetrized = 3

struct CommutatorFreeScheme
    A::Array{Float64,2}
    c::Array{Float64,1}
    p::Int
end

get_order(scheme::CommutatorFreeScheme) = scheme.p
number_of_exponentials(scheme::CommutatorFreeScheme) = size(scheme.A, 1)

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

function prepare_Omega(H, j::Int, f::Function, which::Int, t::Float64, dt::Float64, scheme::CommutatorFreeScheme)
    set_matrix_times_minus_i!(H, which!=for_expv) 
    if which==for_Gamma_d
       g = 0.0
       f1 = sum(scheme.c.*scheme.A[j,:].*f.(t+dt*scheme.c))
    elseif which==for_Gamma_d_symmetrized
       g = 0.0
       f1 = sum((scheme.c-0.5).*scheme.A[j,:].*f.(t+dt*scheme.c))
    else
       g = sum(scheme.A[j,:])
       f1 = sum(scheme.A[j,:].*f.(t+dt*scheme.c)) 
    end   
    (g, f1)
end

function Omega!(w::Array{Complex{Float64},1}, v::Array{Complex{Float64},1}, H, args::Tuple,
                p::Int, scheme::CommutatorFreeScheme)
    g, f1 = args           
    set_fac!(H, g, f1)
    A_mul_B!(w, H, v) 
end

abstract type Magnus2 end

get_order(::Type{Magnus2}) = 2
number_of_exponentials(::Type{Magnus2}) = 1

function prepare_Omega(H, j::Int, f::Function, which::Int, t::Float64, dt::Float64, scheme::Type{Magnus2})
    set_matrix_times_minus_i!(H, which!=for_expv) 
    g = which==for_Gamma_d?0.0:1.0 
    f1 = f(t+0.5*dt)
    (g, f1)
end

function Omega!(w::Array{Complex{Float64},1}, v::Array{Complex{Float64},1}, H, args::Tuple,
                p::Int, ::Type{Magnus2})
    g, f1 = args
    set_fac!(H, g, f1)
    A_mul_B!(w, H, v)
end


abstract type Magnus4 end

function get_lwsp_liwsp_expv(H, scheme::Type{Magnus4}, m::Integer=30) 
    (lw, liw) = get_lwsp_liwsp_expv(size(H, 2), m)
    (lw+size(H, 2), liw)
end

get_order(::Type{Magnus4}) = 4
number_of_exponentials(::Type{Magnus4}) = 1

function prepare_Omega(H, j::Int, f::Function, which::Int, t::Float64, dt::Float64, scheme::Type{Magnus4})
    set_matrix_times_minus_i!(H, which!=for_expv) 
    c = dt*sqrt(3)/12*1im
    g = which==for_Gamma_d?0.0:1.0 
    f1 = f(t+dt*(1/2-sqrt(3)/6))
    f2 = f(t+dt*(1/2+sqrt(3)/6))
    n = size(H, 2)
    s = unsafe_wrap(Array, pointer(get_wsp(H), length(get_wsp(H))-n+1), n, false)
    (c, g, f1, f2, s)
end

abstract type DoPri45 end

get_lwsp_liwsp_expv(H, scheme::Type{DoPri45}, m::Integer=30) = (7*size(H,2), 0)

get_order(::Type{DoPri45}) = 4

function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H, f::Function, fd::Function, t::Real, dt::Real,
                 scheme::Type{DoPri45};
                 symmetrized_defect::Bool=false)
      c = [0.0 1/5 3/10 4/5 8/9 1.0 1.0]
      A = [0.0         0.0        0.0         0.0      0.0          0.0     0.0
           1/5         0.0        0.0         0.0      0.0          0.0     0.0
           3/40        9/40       0.0         0.0      0.0          0.0     0.0
           44/45      -56/15      32/9        0.0      0.0          0.0     0.0
           19372/6561 -25360/2187 64448/6561 -212/729  0.0          0.0     0.0
           9017/3168  -355/33     46732/5247  49/176  -5103/18656   0.0     0.0
           35/384      0.0        500/1113    125/192 -2187/6784    11/84   0.0]
     # e = [51279/57600 0.0        7571/16695  393/640 -92097/339200 187/2100 1/40]
       e = [71/57600    0.0       -71/16695    71/1920  -17253/339200 22/525 -1/40]    
      n = size(H, 2)
      K = [unsafe_wrap(Array, pointer(get_wsp(H), j*n+1), n, false) for j=1:8]
      s = K[8]
      for l=1:7
          s[:] = psi
          for j=1:l-1
              if A[l,j]!=0.0
                  s[:] += (dt*A[l,j])*K[j][:]
              end
          end
          set_matrix_times_minus_i!(H, true) 
          set_fac!(H, 1.0, f(t+c[l]*dt))
          A_mul_B!(K[l], H, s)
      end
      psi[:] = s[:]
      s[:] = 0.0
      for j=1:7
          if e[j]!=0.0
              s[:] += (dt*e[j])*K[j][:]
          end
      end
      A_mul_B!(psi_est, H, s)
      #psi_est[:] -= psi[:]
      # TODO: K[7] can be reused as K[1] for the next step (FSAL, first same as last)
end

function Omega!(w::Array{Complex{Float64},1}, v::Array{Complex{Float64},1}, H, args::Tuple,
                p::Int, ::Type{Magnus4})
    c, g, f1, f2, s = args

    set_fac!(H, g, f1)
    A_mul_B!(s, H, v)

    w[:] = 0.5*s[:]
    
    set_fac!(H, g, f2)
    A_mul_B!(s, H, s)
    
    w[:] -= c*s

    set_fac!(H, g, f2)
    A_mul_B!(s, H, v)

    w[:] += 0.5*s

    set_fac!(H, g, f1)
    A_mul_B!(s, H, s)
    
    w[:] += c*s
end


function step!(psi::Array{Complex{Float64},1}, H, f::Function, 
               t::Real, dt::Real, scheme)
    state = save_state(H)
    p = get_order(scheme)
    for j=1:number_of_exponentials(scheme)
        args = prepare_Omega(H, j, f, for_expv, t, dt, scheme)
        expv!(psi, dt, Omega!, psi, get_norm0(H),
              args = (H, args, p, scheme),
              matrix_times_minus_i=true, hermitian=true,
              wsp=get_wsp(H), iwsp=get_iwsp(H))
    end
    restore_state!(H, state)
end    

function Gamma!(r::Vector{Complex{Float64}}, H,
                u::Vector{Complex{Float64}}, p::Int, dt::Float64, 
                args::Tuple, argsd::Tuple,
                s1::Vector{Complex{Float64}}, s2::Vector{Complex{Float64}}, scheme)
    #s2=B*u
    Omega!(s2, u, H, args, p, scheme)
    r[:] = s2[:]    
    
    if p>=1
        #s1=A*u
        Omega!(s1, u, H, argsd, p, scheme)
        r[:] += dt*s1[:] 
    end
    if p>=2
        #s1=B*s1=BAu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] += (dt^2/2)*s1[:] 
    end
    if p>=3
        #s1=B*s1=BBAu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] += (dt^3/6)*s1[:] 
    end
    if p>=4
        #s1=B*s1=BBBAu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] += (dt^4/24)*s1[:] 
    end
    if p>=5
        #s1=B*s1=BBBBAu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] += (dt^5/120)*s1[:] 
    end
    if p>=6
        #s1=B*s1=BBBBBAu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] += (dt^6/720)*s1[:] 
    end

    if p>=2
        #s1=A*s2=ABu
        Omega!(s1, s2, H, argsd, p, scheme)
        r[:] -= (dt^2/2)*s1[:] 
    end
    if p>=3
        #s1=B*s1=BABu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] -= (dt^3/3)*s1[:] 
    end
    if p>=4
        #s1=B*s1=BBABu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] -= (dt^4/8)*s1[:] 
    end
    if p>=5
        #s1=B*s1=BBBABu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] -= (dt^5/30)*s1[:] 
    end
    if p>=6
        #s1=B*s1=BBBBABu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] -= (dt^6/144)*s1[:] 
    end

    if p>=3
        #s2=B*s2=BBu
        Omega!(s2, s2, H, args, p, scheme)
        #s1=A*s2=ABBu
        Omega!(s1, s2, H, argsd, p, scheme)
        r[:] += (dt^3/6)*s1
    end
    if p>=4
        #s1=B*s1=BABBu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] += (dt^4/8)*s1
    end
    if p>=5
        #s1=B*s1=BBABBu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] += (dt^5/20)*s1
    end
    if p>=6
        #s1=B*s1=BBBABBu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] += (dt^6/72)*s1
    end

    if p>=4
        #s2=B*s2=BBBu
        Omega!(s2, s2, H, args, p, scheme)
        #s1=A*s2=ABBBu
        Omega!(s1, s2, H, argsd, p, scheme)
        r[:] -= (dt^4/24)*s1
    end
    if p>=5
        #s1=B*s1=BABBBu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] -= (dt^5/30)*s1
    end
    if p>=6
        #s1=B*s1=BBABBBu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] -= (dt^6/72)*s1
    end

    if p>=5
        #s2=B*s2=BBBBu
        Omega!(s2, s2, H, args, p, scheme)
        #s1=A*s2=ABBBBu
        Omega!(s1, s2, H, argsd, p, scheme)
        r[:] += (dt^5/120)*s1
    end
    if p>=6
        #s1=B*s1=BABBBBu
        Omega!(s1, s1, H, args, p, scheme)
        r[:] += (dt^6/144)*s1
    end

    if p>=6
        #s2=B*s2=BBBBBu
        Omega!(s2, s2, H, args, p, scheme)
        #s1=A*s2=ABBBBBu
        Omega!(s1, s2, H, argsd, p, scheme)
        r[:] -= (dt^6/720)*s1
    end
end


function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H, f::Function, fd::Function, t::Real, dt::Real,
                 scheme::CommutatorFreeScheme;
                 symmetrized_defect::Bool=false)
    state = save_state(H)

    p = get_order(scheme)
    for_Gamma_d1 = symmetrized_defect?for_Gamma_d_symmetrized:for_Gamma_d

    n = size(H, 2)
    s = unsafe_wrap(Array, pointer(get_wsp(H), 1), n, false)
    s1 = unsafe_wrap(Array, pointer(get_wsp(H), n+1),   n, false)
    s2 = unsafe_wrap(Array, pointer(get_wsp(H), 2*n+1), n, false)

    if symmetrized_defect
        set_fac!(H, -0.5, -0.5*f(t))
        set_matrix_times_minus_i!(H, true) 
        LinAlg.A_mul_B!(psi_est, H, psi)

        args = prepare_Omega(H, 1, f, for_expv, t, dt, scheme)
        expv!(psi_est, dt, Omega!, psi_est, get_norm0(H), 
              args = (H, args, p, scheme),
              matrix_times_minus_i=true, hermitian=true,
              wsp=get_wsp(H), iwsp=get_iwsp(H)) 
    else
        psi_est[:] = 0.0
    end

    args = prepare_Omega(H, 1, f, for_expv, t, dt, scheme)
    expv!(psi, dt, Omega!, psi, get_norm0(H), 
          args = (H, args, p, scheme),
          matrix_times_minus_i=true, hermitian=true,
          wsp=get_wsp(H), iwsp=get_iwsp(H))

    args  = prepare_Omega(H, 1, f,  for_Gamma,    t, dt, scheme)
    argsd = prepare_Omega(H, 1, fd, for_Gamma_d1, t, dt, scheme)
    Gamma!(s, H, psi, p, dt, args, argsd, s1, s2, scheme)
    psi_est[:] += s[:]

    for j=2:size(scheme.A, 1)

        args = prepare_Omega(H, j, f, for_expv, t, dt, scheme)
        expv!(psi_est, dt, Omega!, psi_est, get_norm0(H), 
              args = (H, args, p, scheme),
              matrix_times_minus_i=true, hermitian=true,
              wsp=get_wsp(H), iwsp=get_iwsp(H))

        args = prepare_Omega(H, j, f, for_expv, t, dt, scheme)
        expv!(psi, dt, Omega!, psi, get_norm0(H), 
              args = (H, args, p, scheme),
              matrix_times_minus_i=true, hermitian=true,
              wsp=get_wsp(H), iwsp=get_iwsp(H))

        args  = prepare_Omega(H, j, f,  for_Gamma,    t, dt, scheme)
        argsd = prepare_Omega(H, j, fd, for_Gamma_d1, t, dt, scheme)
        Gamma!(s, H, psi, p, dt, args, argsd, s1, s2, scheme)
        psi_est[:] += s[:]

    end
   
    if symmetrized_defect
        set_fac!(H, 0.5, 0.5*f(t+dt))
    else
        set_fac!(H, 1.0, f(t+dt))
    end
    LinAlg.A_mul_B!(s, H, psi)
    psi_est[:] -= s[:]

    psi_est[:] *= dt/(p+1)

    restore_state!(H, state)
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
    lwsp1, liwsp1 = get_lwsp_liwsp_expv(H, scheme)  
    lwsp2, liwsp2 = get_lwsp_liwsp_expv(H, reference_scheme)  
    set_wsp!(H, max(lwsp1, lwsp2))
    set_iwsp!(H, max(liwsp1, liwsp2))

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
    lwsp1, liwsp1 = get_lwsp_liwsp_expv(H, scheme)  
    lwsp2, liwsp2 = get_lwsp_liwsp_expv(H, reference_scheme)  
    set_wsp!(H, max(lwsp1, lwsp2))
    set_iwsp!(H, max(liwsp1, liwsp2))

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
    dt_max::Float64
    scheme
    psi_est::Array{Complex{Float64},1}
    psi0::Array{Complex{Float64},1}

    function AdaptiveTimeStepper(H, 
                 f::Function, fd::Function,
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real,  tol::Real; scheme=CF4, dt_max=realmax(Float16))
        order = get_order(scheme)

        # allocate workspace
        lwsp, liwsp = get_lwsp_liwsp_expv(H, scheme)  
        set_wsp!(H, lwsp)
        set_iwsp!(H, liwsp)

        psi_est = zeros(Complex{Float64}, size(H, 2))
        psi0 = zeros(Complex{Float64}, size(H, 2))
        
        new(H, f, fd, psi, t0, tend, dt, tol, order, dt_max, scheme, psi_est, psi0)
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
        dt = min(ats.dt_max, dt*min(facmax, max(facmin, fac*(1.0/err)^(1.0/(ats.order+1)))))
        if err>=1.0
           ats.psi[:] = ats.psi0
           @printf("t=%17.9e  err=%17.8e  dt=%17.8e  rejected...\n", Float64(state.t), Float64(err), Float64(dt))
        end   
    end
    state.t + dt0, AdaptiveTimeStepperState(state.t+dt0, dt)
end



