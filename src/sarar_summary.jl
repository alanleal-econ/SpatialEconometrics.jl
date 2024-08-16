using Optim
using LinearAlgebra
using ForwardDiff
using Statistics
using Distributions
using Printf
function sarar_summary(sarmodel,names_col)
    print(".------------.---------.---------.----------.----------.-----------.","\n")
    print("Maximum Likelihood Estimation of SAC Model","\n")
    print(".------------.---------.---------.----------.----------.-----------.","\n")
    print("Log-Likelihood: ",round(sarmodel.ll,digits=3),"\n")
    print("Number of observations: ",sarmodel.nobs,"\n")
    print("σ2: ",round(sarmodel.sigma2,digits=3),"\n")
    names_columns=vcat(names_col)
    df_definitive=hcat(names_columns,sarmodel.coefs)
    header = ["Variable","β", "Std Dev", "Lower CI", "Upper CI", "p-value"]
    pretty_table(df_definitive, header=header, alignment = :c, formatters = ft_printf("%.3f", 1:5),tf=tf_ascii_rounded)
    print("λ: ", round(sarmodel.rho[1],digits=3), ", Standard Deviation: ", round(sarmodel.rho[2],digits=3), "\n")
    print("λ: ", round(sarmodel.lambda[1],digits=3), ", Standard Deviation: ", round(sarmodel.lambda[2],digits=3), "\n")
end