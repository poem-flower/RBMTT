clc
clear

T=0.1;
V_min =0;
V_max = 340;
R_min = 0;
R = diag([200 200]);
Q = diag([0.05 0.05 1 1].^2);
H=[1 0 0 0;
       0 1 0 0];
alpha_seq = (-10:0.1:10)*pi/180;



% mode=0 generate data for training；
% mode=1 generate data for testing
mode = 1;
if mode==0
    M = 200000;
    N=20;
    filepath = "./train_data/";
else
    M = 500;
    N=100;
    filepath = "./test_data/";
end

R_max = 37040 - V_max*N*T;
X = zeros(M, N, 4);
Z_cart = zeros(M, N, 2);

tic
parfor k = 1:M
    F = gen_F(N, T, alpha_seq, mode);
    x_temp = zeros(4, N);
    x_temp(:, 1) = gen_initial(R_min, R_max, V_min, V_max);

    for t=2:N
        x_temp(:, t) = F{1, t}*x_temp(:, t-1)+sqrt(Q)*randn(4, 1);
    end
    X(k, :, :) = x_temp';
    Z_cart(k, :, :) = transpose(H*x_temp+sqrt(R)*randn(2, N));

    disp(['Track generated: ' num2str(k)])

end
toc

%% 保存数据
save(strcat(filepath, 'true_cart'), "X")
save(strcat(filepath, 'measure_cart'), "Z_cart")
