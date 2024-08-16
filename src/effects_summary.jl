using Optim
using LinearAlgebra
using ForwardDiff
using Statistics
using Distributions
using Printf
function effects_summary(efeitos,names_cols)
    df_definitive=hcat(names_cols,efeitos)
    header = ["Variable","Direct Effects", "Indirect Effects", "Total Effects"]
    pretty_table(df_definitive, header=header, alignment = :c, formatters = ft_printf("%.3f", 1:7),tf=tf_ascii_rounded)
end