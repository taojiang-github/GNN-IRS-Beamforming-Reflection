close all
clear;
addpath('./WSM_util/')
load('./channel_data/channel(8, 100, 3)_10.mat');
[num_sample,M,N] = size(channel_bs_irs);
[~,~,K] = size(channel_bs_user);

etai=1;
path_d = 1;
weight=1./((path_d));
weight=weight./sum(weight);
omega=weight;
%%
snr=10;
Pt=10.^(snr/10);
it=1000;
rate_w=zeros(num_sample,it);
start=zeros(1,num_sample);
error_Ellipsoid=0;
beamforming_error=0;
armijo_w=zeros(num_sample,it);
ts = tic();
w_all = zeros(num_sample,M,K);
theta_all = zeros(num_sample,N,1);
rate_perfect = 0;
parfor t0=1:num_sample
    rate_tmp=nan(1,it);
    W_old=zeros(M,K);
    beta=zeros(1,K);
    %%
    Hd=squeeze(channel_bs_user(t0,:,:))'; %%%%%%%%%%%%%%%%%%%%%%%%%%
    theta=exp(1j.*rand(N,1).*2.*pi);
    %     theta=theta_init(:,:,t0);
    Theta=diag(theta');
    %%
    G=squeeze(channel_bs_irs(t0,:,:))';
    Hr=squeeze(channel_irs_user(t0,:,:))';
    %Hd, Hr should be matrix (num_user, num_antenna)
    if K ==1
        Hd = Hd.';
        Hr = Hr.';
    end
    %%
    
    H=Hd+Hr*Theta*G; %%%%%%%%%%%%%%%%%%%%%%%channel%%%%%%%%%%%%%%%
    
    %% init
    [W,grt,f0 ] = init_W( H,M,K,Pt,omega );
    start(t0)=f0;
    f1=0;
    %%
    W_span=W;
    W_last=W;
    [ ~,L_last ] = Proxlinear_beam_para( H,K,M,beta );
    %%
    flag=0;
    t_old=1;
    [ beta ] =upadte_beta( H,W,K,grt);
    for con0=1:it
        [ Qx,qx,theta ] = surface_U_v_direct( W,Hd,Hr,Theta,G,N,K,grt,beta );
        theta_old=theta;
        %%
        U=-Qx;v=qx;
        x0=theta_old;
        phi0=angle(x0);
        grad=real((2*U*x0-2*v).*(-1j.*conj(x0)));
        dir=-grad;
        [ Ltheta ] = SCA_phi_step_para( -Qx,qx,N, theta );
        [ theta,t3,armijo_w(t0,con0) ] = armijo_theta( Ltheta,dir,f0,phi0, grad,grt,W,W_span,t_old,L_last,K,M,Pt,omega,Hd,Hr,G);
        %%
        [f1,grt,beta,W,W_span,t_old,L_last,H,Theta ] = fun_theta_package(grt,theta,W,W_span,t_old,L_last,K,M,Pt,omega,Hd,Hr,G);
        %%
        if  con0>2 && rate_tmp(con0)-rate_tmp(con0-1)<1e-3
            break;
        end
        rate_tmp(con0)=f1;
        f0=f1;
    end
    rate_w(t0,:)=rate_tmp;
    w_all(t0,:,:) = W;
    %      theta = diag(Theta');
    rate_perfect = rate_perfect + rate_tmp(con0);
    theta_all(t0,:,:) = theta;
end
rate_w=mean(rate_w);
start=mean(start);
rate_w=[start rate_w];
rate_perfect = rate_perfect/num_sample
run_time = toc(ts)
%%
fprintf('error Ellipsoid=%d\n',error_Ellipsoid);
fprintf('error beamforming=%d\n',beamforming_error);
%%
figure
plot(0:it,rate_w,'r-');
% save('converge_speed_step_phi_CG_n5.mat','it','rate_w','M','K','N','omega');
%%
armijo_w=mean(armijo_w);
figure
plot(1:it, armijo_w,'r-');
grid on
%%
str = sprintf('./results_data/bcd_perfect_csi_(8, 100, 3)_10_%d.mat',snr);
save(str,'w_all','theta_all','rate_w','run_time','num_sample',...
'rate_perfect');
% run main_imperfect_csi.m


