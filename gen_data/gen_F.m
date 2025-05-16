function F=gen_F(N, T, alpha_seq,mode)
%生成目标状态转移矩阵的矩阵
%N:采样点数
%T:采样间隔
%alpha:最大转弯率的模
%mode==0生成训练数据；mode==1生成测试数据

if (mode==0)
    alpha=randsample(alpha_seq, 1);   %随机抽取转弯率
    F = repmat({F_(alpha, T)}, 1, N);

else
    change_id = sort(randsample(N, 2)); %随机生成目标运动状态变换的时刻点，不允许重复
    label = randi([0 1], 1, 3);  %生成决定目标在每个阶段做CT/CV运动的标签，CV和CT的概率各为1/2
    alpha = label.*randsample(alpha_seq, 3);  %生成每个阶段的转弯率

    F = repmat({F_(alpha(1), T)}, 1, change_id(1));
    F = [F repmat({F_(alpha(2), T)}, 1, change_id(2)-change_id(1))];
    F =[F repmat({F_(alpha(3), T)}, 1,N-change_id(2))];


end


end

