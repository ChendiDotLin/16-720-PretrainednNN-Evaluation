% function[] = bar_visualize()
clc; clear all; close all;
resnet = load('resnet.mat');
nets = {};
nets{end+1} = resnet;
names = {'resnet'};
num_nets = length(names);
epochs = length(resnet.epoch);
% figure();
for i = 1:epochs
    %     train_loss_data = [];
    %     train_acc_data = [];
    %     test_loss_data = [];
    %     test_acc_data = [];
    %     time_data = [];
    for j = 1:num_nets
        net = nets{j};
        train_loss_data(j) =net.train_loss(i);
        %         train_acc_data(j) = net.train_acc(i);
        %         train_acc_data(j) = net.train_acc(i);
        %         test_loss_data(j) = net.test_loss(i);
        %         test_acc_data(j) = net.test_acc(i);
        %         time_data(j) = net.time(i);
        
        
    end
    [data,index] = sort(train_loss_data);
%     [data,index] = sort(train_acc_data);
%     [data,index] = sort(test_loss_data);
%     [data,index] = sort(test_acc_data);
%     [data,index] = sort(time_data);

    barh(data);    
    set(gca,'yticklabel',names(index))
    set(gca,'XLim',[0,5])
    title('train loss comparison')
    pause(0.01)
    M(i) = getframe(gcf);
end
v = VideoWriter('train_loss_comparison.avi');
open(v);
writeVideo(v,M);
close(v);


