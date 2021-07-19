clear;close all
% load 'array_response_irs[[ 30   5 -20]].mat'
load 'array_response_irs[ 10  -5 -20  10   5 -20  10  15 -20].mat'
%% plot 2d array response
figure;
imagesc(set_phi_2,set_phi_3,array_response_irs1)
xlabel('$\phi_3$','Interpreter','latex')
ylabel('$\phi_2$','Interpreter','latex')
colorbar
% matlab2tikz('array_response_2d.tex')

figure;
imagesc(set_theta_3, set_phi_3,array_response_irs2)
xlabel('$\phi_3$','Interpreter','latex')
ylabel('$\theta_3$','Interpreter','latex')
colorbar

figure;
surf(set_theta_3,set_phi_3,array_response_irs2,'EdgeColor','flat')
colorbar
xlabel('$\phi_3$','Interpreter','latex')
ylabel('$\theta_3$','Interpreter','latex')

indx = find(set_phi_3<phi_true(3));
ZmaxCol = indx(end)+1;
indx = find(set_theta_3<irs_theta(2));
ZmaxRow =  indx(end)+1;
hold on
scatter3(set_phi_3(ZmaxCol),set_theta_3(ZmaxRow),array_response_irs2(ZmaxRow,ZmaxCol),'filled')
%% plot 1d array response
phi_true(2)
phi_true(3)
figure;
indx = find(set_phi_2<phi_true(2));
plot(set_phi_3,array_response_irs1(indx(end)+1,:),'LineWidth',2)
grid on
xlabel('$\phi_3$','Interpreter','latex')
ylabel('Array response','Interpreter','latex')
% matlab2tikz('array_response_1d.tex')


%% plot 1d array response
phi_true(3)
figure;
indx = find(set_phi_3<phi_true(3));
plot(set_theta_3,array_response_irs2(:,indx(end)+1),'LineWidth',2)
grid on
xlabel('$\theta_3$','Interpreter','latex')
ylabel('Array response','Interpreter','latex')
% matlab2tikz('array_response_1d.tex')

%% plot array response to BS
figure;
load 'array_response_bs[[ 30  20 -20]].mat'
% load 'array_response_bs[[ 30  20 -20]].mat'

plot(set_phi_1,array_response_bs,'LineWidth',2)
grid on
xlabel('$\phi_1$','Interpreter','latex')
ylabel('Array response','Interpreter','latex')
% 
% figure;
% imagesc(set_theta_1_2d,set_phi_1_2d,array_response_bs_2d)
% xlabel('$\theta_1$','Interpreter','latex')
% ylabel('$\phi_1$','Interpreter','latex')
% colorbar
