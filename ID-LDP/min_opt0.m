function [X,FVAL] = min_opt0(W,fun)
N_loc = length(W);

c = @(x) nonlcon(x,N_loc,W);


LB = [0.5*ones(N_loc,1); zeros(N_loc,1)]; #semi colon are new rows
UB = [ones(N_loc,1); 0.5*ones(N_loc,1)];


x0 = [0.5*ones(N_loc,1); 1/(1+exp(min(W)))*ones(N_loc,1)];

options = optimoptions('fmincon','Algorithm','sqp');
[X,FVAL,EXITFLAG] = fmincon(fun,x0,[],[],[],[],LB,UB,c,options);

end



function [c, ceq] = nonlcon(x,N_loc,W) %anonymous fn defining constraints c(x) <= 0, and ceq(x) = 0 
% https://www.mathworks.com/help/optim/ug/nonlinear-constraints.html


row = @(i,j) N_loc*(i-1) + j; % row is an anonymous function on i,j just mapping to an index starting at 1,1 -> 1 for matlan indexing

a = x(1:N_loc); b = x(N_loc+1:end);

c = zeros(N_loc^2,1);
for i = 1:N_loc
    for j = 1:N_loc
        c(row(i,j)) = a(i)*(1-b(j))-exp(min(W(i),W(j)))*b(i)*(1-a(j));
    end
end

ceq = [];
end