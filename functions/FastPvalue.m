function p = FastPvalue(obs,null_distr,tail)

p=sum(null_distr>obs)/length(null_distr);

if tail==-1
    p=1-p;
elseif tail==2
    p(p>=.5) = 1-p(p>=.5);
    p = p*2;
end
end