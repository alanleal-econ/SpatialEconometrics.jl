using Optim
using LinearAlgebra
using ForwardDiff
using Statistics
using Distributions
using Printf
using PrettyTables
function effects_sar(y,X,W,β1)
    n=length(y)
    ψ=inv(I(n)-β1[2]*W)
    eft_direto=zeros(length(β1)-3)
    eft_indireto=zeros(length(β1)-3)
    eft_total=zeros(length(β1)-3)
    for i=4:length(β1)
        eft_direto[i-3]=(1/n)*tr(inv(I(n)-β1[2]*W))*β1[i]
        eft_total[i-3]=(1/n)*sum(sum(ψ[k,j]*β1[i] for j=1:n) for k=1:n)
        eft_indireto[i-3]=eft_total[i-3]-eft_direto[i-3]
    end
    return hcat(eft_direto,eft_indireto,eft_total)
end  