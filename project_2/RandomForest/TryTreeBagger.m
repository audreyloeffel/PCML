BaggedEnsemble = TreeBagger(50,train.X_hog,train.y,'OOBPred','On');
oobErrorBaggedEnsemble = oobError(BaggedEnsemble);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';