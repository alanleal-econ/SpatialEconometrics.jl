using Optim
using LinearAlgebra
using ForwardDiff
using Statistics
using Distributions
using Printf
using PrettyTables
function sar_likelihood(params,n,X,y,W)
    σ2,ρ, β = params[1],params[2], params[3:end]
    ε = (I(n) - ρ*W)*y-X*β
    loglik = -(n/2)*log(2*pi*σ2)-((1/(2*σ2))*(ε'*ε))+logdet(I(n)-ρ*W)
    return -loglik
end
function sar_coef(X,y,W)
    n_x=size(X)[2]
    n=size(X)[1]
    initial_params = vcat(10,0.5,zeros(n_x))
    lower_bounds = [0;-1;fill(-Inf,n_x)]
    upper_bounds = [Inf;1;fill(Inf,n_x)]
    result = optimize(params -> sar_likelihood(params, n,X,y,W),lower_bounds, upper_bounds,initial_params,Fminbox())
    β = result.minimizer
    ll=-result.minimum
    return  β,ll
end

function sar_sdev(X,y,W,β,n)
    likelihood_β_only = β -> sar_likelihood(β, n, X, y, W)
    H=ForwardDiff.hessian(likelihood_β_only, β)
    σk = ForwardDiff.value.(real.(H))\I
    desvio_padrao=sqrt.(diag(σk))
    return desvio_padrao
end

function sar_pvalor(y,X,coefs,desvios_padroes)
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

function sar(X,y,W)
    n=size(X)[1]
    coefs,ll=sar_coef(X,y,W)
    desvios_padroes=sar_sdev(X,y,W,coefs,n)
    ic_pvalores=sar_pvalor(y,X,coefs,desvios_padroes)
    sigma2=coefs[1]
    rho=vcat(coefs[2],desvios_padroes[2])
    coefs=hcat(coefs,desvios_padroes,ic_pvalores)[3:end,:]
    nobs=n
    k=length(coefs[:,1])
    dof=n-k
    results=(coefs = coefs, sigma2 = sigma2, rho = rho,nobs=nobs,dof=dof,ll=ll)
    return results
end