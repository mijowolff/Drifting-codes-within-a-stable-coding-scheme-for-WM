 function p = GroupPermTest(dat, nSims,tail,diffstat)
%% test dat against zero
%% dat can be any number of dimensions. Group must be the first dimension

if nargin < 4 || isempty(diffstat)
    diffstat = 'diff';
end
usetscore = strcmp(diffstat, 't');

nSamp = size(dat,1);
sqrtNSamp = sqrt(nSamp);

mu=mean(dat,1);
if usetscore
    sd=std(dat,[],1);
    mdat=mu./(sd./sqrtNSamp);
else
    mdat = mean(dat,1);
end
p = zeros(size(mdat));
for sim=1:nSims
    permind = sign(rand(size(dat,1),1)-.5);
    permind = repmat(permind,[size(mdat)]);
    mu_tmp=mean(dat.*permind,1);
    if usetscore
        sd = std(dat.*permind, [],1);
        mrnd=mu_tmp./(sd./sqrtNSamp);
    else
        mrnd=mu_tmp;
    end
    p = p + (mrnd>=mdat);
end
p = p./nSims;
if p==0
    p=1/nSims;
end
if tail==-1
    p=1-p;
elseif tail==2
    p(p>=.5) = 1-p(p>=.5);
    p = p*2;
end
end