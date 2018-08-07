function [ cum_wealth, daily_incre_fact, daily_port_total ] = KTPT_run(data)
%{
This function is the main code for the A kernel-based trend pattern tracking system 
for portfolio optimization (KTPT)[5]system. It includes a three-state price prediction scheme, 
which extracts both of the following and reverting patterns from the asset price trend 
to make future price predictions. Moreover, KTPT is equipped with a novel kernel-based tracking
system to optimize the portfolio, so as to capture a potential growth of the asset price effectively.

For any usage of this function, the following papers should be cited as
reference:

[1] Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren, and Ke-Kun Huang. ¡°A peak price tracking 
based learning system for portfolio selection¡±, 
IEEE Transactions on Neural Networks and Learning Systems, 2017. Accepted.
[2] Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren, and Ke-Kun Huang.  ¡°Radial basis functions 
with adaptive input and composite trend representation for portfolio selection¡±, 
IEEE Transactions on Neural Networks and Learning Systems, 2018. Accepted.
[3] Pei-Yi Yang, Zhao-Rong Lai*, Xiaotian Wu, Liangda Fang. ¡°Trend Representation 
Based Log-density Regularization System for Portfolio Optimization¡±,  
Pattern Recognition, vol. 76, pp. 14-24, Apr. 2018.
[4]Zhao-Rong Lai, Pei-Yi Yang,  Liangda Fang and Xiaotian Wu.¡°Reweighted Price Relative 
Tracking System for Automatic Portfolio Optimization¡±. 
IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2018. Accepted.
[5]Zhao-Rong Lai, Pei-Yi Yang, Xiaotian Wu and Liangda Fang. ¡°A kernel-based trend 
pattern tracking system for portfolio optimization¡±, 
Data Mining and Knowledge Discovery, 2018. Accepted.

At the same time, it is encouraged to cite the following papers with previous related works:

[6] J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra, ¡°Efficient
projections onto the \ell_1-ball for learning in high dimensions,¡± in
Proceedings of the International Conference on Machine Learning (ICML 2008), 2008.
[7] B. Li, D. Sahoo, and S. C. H. Hoi. Olps: a toolbox for on-line portfolio selection.
Journal of Machine Learning Research, 17, 2016.

Inputs:
data                      -data with price relative sequences

Outputs:
cum_wealth                -cumulative wealths
daily_incre_fact          -daily increasing factors of KTPT
daily_port_total          -daily selected portfolios of KTPT
%}

%% Parameter Setting
tran_cost=0;    % -transaction cost rate
win_size=5;     % -window size
W2=10;          % -window size to calculate reverting points

%% Variables Inital
[T,N] = size(data);
cum_wealth = ones(T, 1);
daily_incre_fact = ones(T, 1);
daily_port_total=ones(N, T)/N;
daily_port = ones(N, 1)/N;  
daily_port_o = zeros(N, 1);
run_ret=1; 
P_t_hat = [];
close_price = ones(T,N);
for i=2:T
    close_price(i,:)= close_price(i-1,:).*data(i,:);
end
Sign=zeros(T,N);
num_reverse=zeros(T,N);

%% Calculate Trend-Reverting Points in the Observed Window
for i=2:T-1
    for j=1:N
        if ((close_price(i,j)-close_price(i-1,j))*(close_price(i,j)-close_price(i+1,j)))>0
            Sign(i,j)=1;
        else
            Sign(i,j)=0;
        end
    end
end
for i=3:1:W2
    num_reverse(i,:)=sum(Sign(1:i-2,:));
end
for i = W2+1:1:T
    num_reverse(i,:) = sum( Sign((i-W2):i-2,:) );
end

%% Mian 
for t = win_size:1:T
    daily_port_total(:,t) = daily_port;
    daily_incre_fact(t, 1) = (data(t, :)*daily_port)*(1-tran_cost/2*sum(abs(daily_port-daily_port_o)));
    run_ret = run_ret * daily_incre_fact(t, 1);
    cum_wealth(t, 1) = run_ret;
    daily_port_o = daily_port.*data(t, :)'/daily_incre_fact(t, 1);
    if t < T
       [daily_port_n, P_t_hat] = KTPT(close_price, data, P_t_hat, num_reverse, t+1, daily_port, win_size, W2);
       daily_port = daily_port_n;  
    end
end
end

