using Optim
using LinearAlgebra
using ForwardDiff
using Statistics
using Distributions
using Printf
using PrettyTables
function effects_sarar(y,X,W,M,β1)
    n=lenghth(y)
    ψ=inv(I(n)-β1[2]*W)
    eft_direto=zeros(length(β1)-4)
    eft_indireto=zeros(length(β1)-4)
    eft_total=zeros(length(β1)-4)
    for i=5:length(β1)
        eft_direto[i-4]=(1/n)*tr(inv(I(n)-β1[2]*W))*β1[i]
        eft_total[i-4]=(1/n)*sum(sum(ψ[k,j]*β1[i] for j=1:n) for k=1:n)
        eft_indireto[i-4]=eft_total[i-4]-eft_direto[i-4]
    end
    return hcat(eft_direto,eft_indireto,eft_total)
end  
