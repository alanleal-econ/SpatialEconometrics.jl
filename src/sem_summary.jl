using Optim
using LinearAlgebra
using ForwardDiff
using Statistics
using Distributions
using Printf
using PrettyTables
function sem_summary(semmodel,names_col)
    print(".------------.---------.---------.----------.----------.-----------.","\n")
    print("Maximum Likelihood Estimation of SEM Model","\n")
    print(".------------.---------.---------.----------.----------.-----------.","\n")
    print("Log-Likelihood: ",round(semmodel.ll,digits=3),"\n")
    print("Number of observations: ",semmodel.nobs,"\n")
    print("σ2: ",round(semmodel.sigma2,digits=3),"\n")
    names_columns=vcat(names_col)
    df_definitive=hcat(names_columns,semmodel.coefs)
    header = ["Variable","β", "Std Dev", "Lower CI", "Upper CI", "p-value"]
    pretty_table(df_definitive, header=header, alignment = :c, formatters = ft_printf("%.3f", 1:5),tf=tf_ascii_rounded)
    print("λ: ", round(semmodel.lambda[1],digits=3), ", Standard Deviation: ", round(semmodel.lambda[2],digits=3), "\n")
end