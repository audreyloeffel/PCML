function loss = logLoss(y,pred)
   N = length(y);
   loss = 0;
   for i = 1:N
       loss = loss + y(i)* log(pred(i))+(1-y(i))*log(1-pred(i)) ;
   end
   loss = -loss/N;
end