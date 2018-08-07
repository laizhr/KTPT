function [b_tplus1_hat, P_tplus1_hat] = KTPT(close_price, data, P_t_hat, num_reverse, tplus1, b_t_hat, win_size, w2)
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
close_price               -close price sequences
data                      -data with price relative sequences
P_t_hat                   -price prediction at time t
num_reverse               -number of trend-reverting points in the observed window
tplus1                    -t+1
b_t_hat               	  -selected portfolio at time t
win_size                  -window size
w2                        -window size to calculate reverting points

Output:
b_tplus1_hat              -selected portfolio at time t+1
P_tplus1_hat              -price prediction at time t+1
%}

%% Parameter Setting
nu = 0.5;               % -mixing parameter
eta = 1000;             % -step size
sigmasquare = 1/6;      % -shape parameter
%% Main
[~, N] = size(data);
if tplus1 < win_size+2
   x_tplus1_hat = data(tplus1-1,:);
   P_tplus1_hat=x_tplus1_hat.*close_price(tplus1-1,:);
else
   x_t = data(tplus1-1,:);
   P_t = close_price((tplus1-win_size):(tplus1-1),:);
   % initial state
   P_tplus1_tilde = max(P_t);									
   y_tplus1 = nu*P_tplus1_tilde+(1-nu)*P_t_hat;  				
   % intermediate state
   z_t = lasso(P_t', y_tplus1', 'Alpha', 0.99); 				
   jay = size(z_t,2);
   y_tplus1_hat = z_t(:,ceil(jay/2))'*P_t; 						
   % final state
   if tplus1 < w2+2
       lambda = 1/N*(sum(num_reverse(tplus1,:))/(tplus1-2));	
   else
       lambda = 1/N*(sum(num_reverse(tplus1,:))/(w2-1));		
   end
   P_tplus1_hat = lambda.*(1./(2*x_t)).*P_tplus1_tilde+(1-lambda.*(1./(2*x_t))).*y_tplus1_hat; 
   x_tplus1_hat = P_tplus1_hat./close_price(tplus1-1,:);		
end
x_tplus1_hat = x_tplus1_hat';
x_tplus1_tilde = (x_tplus1_hat-mean(x_tplus1_hat));
b_t_tilde = b_t_hat-mean(b_t_hat);

K_t = exp(-abs(b_t_tilde-x_tplus1_tilde).^sigmasquare);			

b_tplus1_hat = b_t_hat + eta*diag(K_t)*x_tplus1_tilde;			
b_tplus1_hat = simplex_projection_selfnorm2(b_tplus1_hat,1);
end