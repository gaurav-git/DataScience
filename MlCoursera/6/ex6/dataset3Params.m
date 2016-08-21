function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
Cz = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmaz = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
best = [0; 0];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
prevError = 100;
no = 0;
error = zeros(length(Cz),length(sigmaz));
for i = 1: length(Cz)
	for j = 1: length(sigmaz)

		no +=1;
		model= svmTrain(X, y, Cz(i), @(x1, x2) gaussianKernel(x1, x2, sigmaz(j)));
		predictions = svmPredict(model, Xval);
		error(i,j) = mean(double(predictions ~= yval));

		if error(i,j)< prevError 
			i;
			j;
			best = [Cz(i);sigmaz(j)];
			prevError = error(i,j);
		end

		

	end
end	
error;
C= best(1);
sigma = best(2);




% =========================================================================

end
