function F = F_(alpha, T)
%生成目标状态转移矩阵
if (alpha==0)
    F = [1 0 T 0;
        0 1 0 T;
        0 0 1 0;
        0 0 0 1];     %CV模型
else
    F = [1 0 sin(alpha*T)/alpha (cos(alpha*T)-1)/alpha;
               0 1 (1-cos(alpha*T))/alpha sin(alpha*T)/alpha;
               0 0 cos(alpha*T) -sin(alpha*T);
               0 0 sin(alpha*T) cos(alpha*T)];     %CT模型
end

end