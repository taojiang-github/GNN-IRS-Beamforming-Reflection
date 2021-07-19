clear; close all


blue = [0.00,0.45,0.74];
orange = [0.85,0.33,0.10];
yellow = [0.93,0.69,0.13];
purple = [0.49,0.18,0.56];
green = [0.47,0.67,0.19];
%%
load results_sumrate_num_user_Pd10.mat
num_user_set = num_user_set;
rate_perfectCSI = rate_perfectCSI;
rate_imperfectCSI = rate_imperfectCSI;

load results_test_rate_vs_num_user10.mat
rate_nn = rate_sum;


plot(num_user_set,rate_perfectCSI,...
    'v-','LineWidth',1.5,...
    'Color',yellow,....
    'MarkerSize',7,...
    'MarkerFaceColor',yellow)
hold on

plot(num_user_set,rate_imperfectCSI,...
    's-','LineWidth',1.5,...
    'Color',blue,....
    'MarkerSize',8,...
    'MarkerFaceColor',blue)
hold on

plot(num_user_set,rate_nn,...
    'o-','LineWidth',1.5,...
    'Color',orange,....
    'MarkerSize',6.7,...
    'MarkerFaceColor',orange)
hold on
grid on

legend('Perfect CSI+BCD','LMMSE channel estimation+BCD','Deep learning','Location','southeast','Interpreter','latex')
xlabel('Number of users $K$','Interpreter','latex')
ylabel('Sum rate [bps/Hz]','Interpreter','latex')
% xticks(Pd_set_nn)
