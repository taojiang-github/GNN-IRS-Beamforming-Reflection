function [ Qx,qx,theta ] = surface_U_v_direct_2( W,Hd,A_channel,Theta,N,K,grt,beta )
    ebs=beta;
    B=Hd*W;
    B=B.';
    [M,N,K] = size(A_channel);

     %%
    A=zeros(N,1,K,K);
    for i0=1:K
        for k0=1:K
%          atp=diag(Hr(k0,:))*G*W(:,i0);
         A_channel_k = squeeze(A_channel(:,:,k0));
         atp=A_channel_k'*W(:,i0);

         A(:,:,i0,k0)=atp;
        end
    end
 %%
    theta=diag(Theta');
 %%
    Qx=zeros(N,N);
    for k0=1:K
        for i0=1:K
            Qx=Qx-abs(ebs(k0))^2.*A(:,:,i0,k0)*A(:,:,i0,k0)';
        end
    end
    qx=zeros(N,1);
    for k0=1:K
        tmp=zeros(N,1);
        for i0=1:K
            tmp=tmp+abs(ebs(k0))^2.*(B(i0,k0)')*A(:,:,i0,k0);
        end
        qx=qx+sqrt(grt(k0))*ebs(k0)'*A(:,:,k0,k0)-tmp;
    end 
end

