function [ gr,grt,f0 ] = update_SINR( H,W,K,omega )
gr=zeros(1,K);
% W = normrnd(0,1,4,4)+1j*normrnd(0,1,4,4);
% W = W/norm(W,'fro');
He=abs(H*W).^2;

 %% update gr (SINR) and grt
 for k0=1:K
     tmp=He(k0,:);
     gr(k0)=tmp(k0)/(sum(tmp)-tmp(k0)+1);
 end
 f0=sum(omega.*log2(1+gr));
 grt=omega.*(1+gr);
end

