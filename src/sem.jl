using Optim
using LinearAlgebra
using ForwardDiff
using Statistics
using Distributions
using Printf
using PrettyTables
function log_likelihood_sem(params,n,X,y,W)
    #n = length(y)
    #I = I(n)  # Identity matrix
   #u = y - X * params[3:end]
    log_det = logdet(I(n) - params[2] * W)
    log_likelihood = -n/2 * log(2 * π) - n/2 * log(params[1]) + log_det - (1 / (2 * params[1])) *(y-X*params[3:end])'* ((I(n)-params[2]*W') * (I(n)-params[2]*W))*(y-X*params[3:end])
    return -log_likelihood
end
function sem_coefs(X,y,W)
    n_x=size(X)[2]
    n=size(X)[1]
    initial_params = vcat(1,0.5,zeros(n_x)) # Initial values for sigma, ρ e β
    lower_bounds = [0;-1;fill(-Inf,n_x)]
    upper_bounds = [Inf;1;fill(Inf,n_x)]
    result = optimize(params -> sar_likelihood(params,n,X,y,W),lower_bounds, upper_bounds,initial_params,Fminbox())
    β = result.minimizer
    ll=-result.minimum
    return β,ll
end
function sem_sdev(X,y,W,β,n)
    likelihood_β_only = β -> sar_likelihood(β, n, X, y, W)
    hessian_matrix = ForwardDiff.hessian(likelihood_β_only, β)
    cov_matrix = inv(hessian_matrix)
    std_devs = sqrt.(diag(cov_matrix))
    return std_devs
end

function sem_pvalor(y,X,coefs,desvios_padroes)
    n=size(X)[1]
    k=length(coefs)
    dof = n - k
    confidence_level = 0.95
    t_value = quantile(TDist(dof), 1 - (1 - confidence_level) / 2)
    confidence_intervals_lower=[(coef - t_value * std) for (coef, std) in zip(coefs, desvios_padroes)]
    confidence_intervals_upper=[(coef + t_value * std) for (coef, std) in zip(coefs, desvios_padroes)]
    p_values = [2 * (1 - cdf(TDist(dof), abs(coef / std))) for (coef, std) in zip(coefs, desvios_padroes)]
    return hcat(confidence_intervals_lower,confidence_intervals_upper,p_values)
end 

function sem(X,y,W)
    n=length(y)
    coefs,ll=sem_coefs(X,y,W)
    sigma2=coefs[1]
    desvios_padroes=sem_sdev(X,y,W,coefs,n)
    ic_pvalores=sar_pvalor(y,X,coefs,desvios_padroes)
    λ=hcat(coefs[2],desvios_padroes[2])
    coefs=hcat(coefs,desvios_padroes,ic_pvalores)[3:end,:]
    nobs=n=size(X)[1]
    k=length(coefs[:,1])
    dof=n-k
    results=(coefs = coefs, sigma2 = sigma2, lambda =λ,nobs=nobs,dof=dof,ll=ll)
    return results
end
