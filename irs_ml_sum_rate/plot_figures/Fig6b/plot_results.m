clear; close all;

load results_train_sumrate0.mat
test_loss= test_loss_set;
% load results_train_sumrate0_15.mat
% test_loss= [test_loss;test_loss_set(1,:)];
load results_train_sumrate0_30.mat
test_loss= [test_loss;test_loss_set];
% load results_train_sumrate0_60.mat
% test_loss= [test_loss;test_loss_set(1,:)];
% load results_train_sumrate0_90.mat
% test_loss= [test_loss;test_loss_set(1,:)];
load results_train_sumrate0_120.mat
test_loss= [test_loss;test_loss_set];



figure;
test_loss(test_loss==0) = NaN;

plot(test_loss(1:end,:)','LineWidth',2)
% for ii =1:length(pilot_set)
%     tmp = ['Pilot length = ',num2str(pilot_set(ii))]
%     print(tmp)
% end
legend( 'Pilot length = 3', 'Pilot length = 6', 'Pilot length = 9',...
    'Pilot length = 15', 'Pilot length = 30', 'Pilot length = 60',...
     'Pilot length = 90', 'Pilot length = 120',...
    'Location','southeast','Fontsize',6.8,'Interpreter','latex')
xlabel('Training epochs','Interpreter','latex')
ylabel('Testing sum rate [bps/Hz]','Interpreter','latex')
grid on
 