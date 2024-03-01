function W = constructZ_sparsity(X, beta)

        XtX =  X' *  X;
        Z = (XtX + beta * eye(size(XtX, 1))) \ XtX;
        Z = normc(Z);

        W = project_simplex(Z);
        W = (W + W')/2;

end

