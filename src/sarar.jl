using Optim
using LinearAlgebra
using ForwardDiff
using Statistics
using Distributions
using Printf
using PrettyTables
function log_likelihood_sarar(params,n,X,y,W,M)
    I_n = I(n)
    u = y - params[2] * W * y - X * params[4:end]
    tilde_u = (I_n - params[3] * M) * u
    log_det_W = logdet(I_n - params[2] * W)
    log_det_M = logdet(I_n - params[3] * M)
    quad_form = tilde_u' * tilde_u
    log_likelihood = - (n / 2) * log(2 * π) - (n / 2) * log(params[1]) +
                    log_det_W + log_det_M - (1 / (2 * params[1])) * quad_form
    return -log_likelihood
end
function sarar_coef(y,X,W,M)
    n_x=size(X)[2]
    n=length(y)
    initial_params = vcat(1,0.5,0.5,zeros(n_x)) 
    lower_bounds = [0;-1;-1;fill(-Inf,n_x)]
    upper_bounds = [Inf;1;1;fill(Inf,n_x)]
    result = optimize(params -> log_likelihood_sarar(params, n,X,y,W,M),lower_bounds, upper_bounds,initial_params,Fminbox())
    β = result.minimizer
    ll=-result.minimum
    return β,ll
end
function sarar_std(X,y,W,M,β,n)
    likelihood_β_only = β -> log_likelihood_sarar(β, n, X, y, W,M)
    hessian_matrix = ForwardDiff.hessian(likelihood_β_only, β)
    cov_matrix = inv(hessian_matrix)
    std_devs = sqrt.(diag(cov_matrix))
    return std_devs
end
function sarar_pvalor(y,X,coefs,desvios_padroes)
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
function sarar(X,y,W,M)
    n=length(y)
    coefs,ll=sarar_coef(y,X,W,M)
    desvios_padroes=sarar_std(X,y,W,M,coefs,n)
    ic_pvalores=sar_pvalor(y,X,coefs,desvios_padroes)
    sigma2=coefs[1]
    ρ=hcat(coefs[2],desvios_padroes[2])
    λ=hcat(coefs[3],desvios_padroes[3])
    coefs=hcat(coefs,desvios_padroes,ic_pvalores)[4:end,:]
    nobs=n=size(X)[1]
    k=length(coefs[:,1])
    dof=n-k
    results=(coefs = coefs, sigma2 = sigma2, rho=ρ,lambda =λ,nobs=nobs,dof=dof,ll=ll)
    return results
end
