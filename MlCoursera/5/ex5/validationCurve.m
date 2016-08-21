function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 1 1.2 1.3 1.5 1.6 1.8 2 2.1 2.3 2.34 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.4  3.5 3.7 3.9 4 4.2 4.6   10]';
m = size(X, 1);
mCv = size(Xval,1);
%' You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%
for i = 1:length(lambda_vec)
	lambda = lambda_vec(i);
	theta = trainLinearReg(X,y,lambda);
	
	hypothesis = X*theta;
	cost = (hypothesis-y).^2;

	error_train(i) = 1/2/m * sum(cost);

	hypCV = Xval*theta; %cv : cross-validation
	costCV = (hypCV-yval).^2;
	error_val(i) = 1/2/mCv * sum(costCV);


end	









% =========================================================================

end
