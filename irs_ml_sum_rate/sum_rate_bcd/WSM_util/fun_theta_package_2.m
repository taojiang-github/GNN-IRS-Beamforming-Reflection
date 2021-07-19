function [f1,grt,beta,W,W_span,t_old,L_last,H,Theta ] = fun_theta_package_2(grt,theta,W,W_span,t_old,L_last,K,M,Pt,omega,Hd,A)
    Theta=diag(theta');
%     H=Hd+Hr*Theta*G;
     H = combine_channel(Hd,theta,A);
    [f1,grt,beta,W,W_span,t_old,L_last ] = fun_theta_obj(grt,H,W,W_span,t_old,L_last,K,M,Pt,omega);
end


