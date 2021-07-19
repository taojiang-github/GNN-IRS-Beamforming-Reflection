clear; close all
load results_test_sumrate_Pt_u.mat
Pd_set_nn = set_len_pilot;
rate_set_nn = rate_sum;

load 'results_sumrate_bl_rate_Pu(8, 100, 3)_10_-100_20_5.mat'
Pd_set_bl = Pt_u_set;
rate_imperfectCSI_set = rate_imperfectCSI;
rate_perfectCSI_set = rate_perfectCSI*ones(1,length(Pd_set_bl));


load results_test_sumrate_Pt_u_generalization.mat
rate_set_generalization = rate_sum;

blue = [0.00,0.45,0.74];
orange = [0.85,0.33,0.10];
yellow = [0.93,0.69,0.13];
purple = [0.49,0.18,0.56];
green = [0.47,0.67,0.19];

plot(Pd_set_bl,rate_perfectCSI_set,...
    'v-','LineWidth',1.5,...
    'Color',yellow,....
    'MarkerSize',7,...
    'MarkerFaceColor',yellow)
hold on

plot(Pd_set_bl,rate_imperfectCSI_set,...
    's-','LineWidth',1.5,...
    'Color',blue,....
    'MarkerSize',8,...
    'MarkerFaceColor',blue)
hold on

plot(Pd_set_nn,rate_set_nn,...
    'o-','LineWidth',1.5,...
    'Color',orange,....
    'MarkerSize',6.7,...
    'MarkerFaceColor',orange)
hold on

plot(Pd_set_nn,rate_set_generalization,...
    '<--','LineWidth',1.5,...
    'Color',green,....
    'MarkerSize',6.7,...
    'MarkerFaceColor',green)
hold on


grid on

legend('Perfect CSI+BCD','LMMSE channel estimation+BCD','Deep learning ($P_u^{\rm{train}}=P_u$)','Deep learning ($P_u^{\rm{train}}=15$dBm)', 'Location','southeast','Interpreter','latex')
xlabel('Uplink pilot transmit power $P_u$ [dBm]','Interpreter','latex')
ylabel('Sum rate [bps/Hz]','Interpreter','latex')