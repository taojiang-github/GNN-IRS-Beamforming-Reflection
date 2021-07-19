clear;close all
% load 'array_response_irs[[ 30   5 -20]].mat'
load 'array_response_irs[  5 -12 -20  12   0 -20   5  12 -20].mat'
%% plot 2d array response

%
figure;
imagesc(set_theta_3, set_phi_3_2,array_response_irs2)
xlabel('$\phi_3$','Interpreter','latex')
ylabel('$\theta_3$','Interpreter','latex')
colorbar

figure;
surf(set_theta_3,set_phi_3_2,array_response_irs2,'EdgeColor','flat')
colorbar
xlabel('$\phi_3$','Interpreter','latex')
ylabel('$\theta_3$','Interpreter','latex')

for ii=1:3
    indx = find(set_phi_3_2<phi_true{3}(ii));
    ZmaxCol = indx(end)+1;
    indx = find(set_theta_3<irs_theta{2}(ii));
    ZmaxRow =  indx(end)+1;
    hold on
    scatter3(set_phi_3_2(ZmaxCol),set_theta_3(ZmaxRow),array_response_irs2(ZmaxRow,ZmaxCol),'filled')
    fprintf('theta=%.3f, phi=%.3f\n',irs_theta{2}(ii),phi_true{3}(ii))
end
%% plot 1d array response
% phi_true(2)
% phi_true(3)
% figure;
% indx = find(set_phi_2<phi_true(2));
% plot(set_phi_3,array_response_irs1(indx(end)+1,:),'LineWidth',2)
% grid on
% xlabel('$\phi_3$','Interpreter','latex')
% ylabel('Array response','Interpreter','latex')
% 
% %% plot 1d array response
% thetas =  irs_theta{2}
% for ii=1:3
%     figure;
%     indx = find(set_phi_3_2<phi_true{3}(ii));
%     plot(set_theta_3,array_response_irs2(:,indx(end)+1),'LineWidth',2)
%     grid on
%     xlabel('$\theta_3$','Interpreter','latex')
%     ylabel('Array response','Interpreter','latex')
% end

% matlab2tikz('array_response_1d.tex')

%% plot array response to BS
% figure;
load 'array_response_bs[  5 -12 -20  12   0 -20   5  12 -20].mat'
figure;
plot(set_phi_1(1,:),array_response_bs(1,:),'b-', 'LineWidth',2)
hold on
plot(set_phi_1(2,:),array_response_bs(2,:),'g-', 'LineWidth',2)
hold on
plot(set_phi_1(3,:),array_response_bs(3,:),'r-', 'LineWidth',2)
hold on
grid on
legend('User 1','User 2','User 3')
xlabel('$\phi_1$','Interpreter','latex')
ylabel('Array response','Interpreter','latex')
% 
% figure;
% imagesc(set_theta_1_2d,set_phi_1_2d,array_response_bs_2d)
% xlabel('$\theta_1$','Interpreter','latex')
% ylabel('$\phi_1$','Interpreter','latex')
% colorbar
