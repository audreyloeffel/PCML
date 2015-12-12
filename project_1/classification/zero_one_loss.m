function loss = zero_one_loss(y,class)
    N = length(y);
    loss = 0;
    for i = 1:N
        if (y(i,1) ~= class(i,1))
            loss = loss + 1;
        end
    end
    loss = loss/N;
    
end