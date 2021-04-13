function [X,FVAL,EXITFLAG] = min_opt1(W,alpha, fun)
N_loc = length(W);

% if exist('alpha','var') % # we need to make a defualt alpha arg switch the order so we can pass in alpha and not fun
if exist('fun','var')
    disp("fun exists")
    % we are being called from python and the anonymous function cannot be passed.
    fun = @(x) alpha*(exp(x)./((exp(x)-1).^2)); % symmetric
else
    fun = alpha % we are in matlab and teh functino was passed as an argument
end

row = @(i,j) N_loc*(i-1) + j; 

A = zeros(N_loc^2,N_loc);
b = zeros(N_loc^2,1);
for i = 1:N_loc
    for j = 1:N_loc
        if i == j
            A(row(i,j),i) = 2;
        else
            A(row(i,j),i) = 1;
            A(row(i,j),j) = 1;
        end
        b(row(i,j)) = min(W(i),W(j));
    end
end
LB = zeros(N_loc,1);
x0 = min(W)*ones(N_loc,1)/2;


options = optimoptions('fmincon','Algorithm','sqp');
[X,FVAL,EXITFLAG] = fmincon(fun,x0,A,b,[],[],LB,[],[],options);
disp(FVAL)

end