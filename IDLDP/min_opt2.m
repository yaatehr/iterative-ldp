function [X,FVAL,EXITFLAG] = min_opt2(W, alpha, fun)
N_loc = length(W);

% if exist('alpha','var') % # we need to make a defualt alpha arg switch the order so we can pass in alpha and not fun
if exist('fun','var')
    disp("fun exists")
    % we are being called from python and the anonymous function cannot be passed.
    fun = @(b) alpha*((b-b.^2)./((0.5-b).^2)) + 1; % a = 0.5
else
    fun = alpha % we are in matlab and teh functino was passed as an argument
end

row = @(i,j) N_loc*(i-1) + j; 

A = zeros(N_loc^2,N_loc);
b = ones(N_loc^2,1);
for i = 1:N_loc
    for j = 1:N_loc
        if i == j
            A(row(i,j),i) = 1+exp(W(i));
        else
            A(row(i,j),i) = exp(min(W(i),W(j)));
            A(row(i,j),j) = 1;
        end
    end
end
LB = zeros(N_loc,1);
UB = 0.5*ones(N_loc,1);
x0 = 1/(exp(min(W))+1)*ones(N_loc,1);

options = optimoptions('fmincon','Algorithm','sqp');
[X,FVAL,EXITFLAG] = fmincon(fun,x0,-A,-b,[],[],LB,UB,[],options);
disp(FVAL)


end