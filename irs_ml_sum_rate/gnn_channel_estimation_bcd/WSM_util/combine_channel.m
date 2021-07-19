function H = combine_channel(Hd,theta,A)
[M,N,K] = size(A);
Hr = zeros(M,K);
for k = 1: K
    A_k = squeeze(A(:,:,k));
    Hr(:,k) = A_k*theta ;   
end
H = Hd+Hr;
H = H';
end