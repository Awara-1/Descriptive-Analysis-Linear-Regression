class LinearRegressionUsingGD:

    def __init__(self, X, Y, X_pred):
        self.X = X
        self.Y = Y
        self.X_pred = X_pred

    def performGD(self, X, Y, X_pred):
        
        m = 0
        c = 0
        n = float(len(X)) #Number of elements in X
        n_iterations = 1000
        L = 0.0001
        
        # Perform Gradient Descent
        for i in range(n_iterations): 
            Y_pred = m*X + c  # The current predicted value of Y
            D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
            D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
            m = m - L * D_m  # Update m
            c = c - L * D_c  # Update c
            
        print(m, c)
        
        # make predictions
        Y_pred = m*X_pred + c
        return Y_pred